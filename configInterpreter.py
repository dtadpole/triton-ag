import os
import re
import json
import time
import asyncio
import traceback
import argparse
import yaml
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections.abc import Iterable
import boto3
import requests
import duckdb
from logger import logger
from itertools import groupby
from operator import itemgetter

class ConfigInterpreter:

    def __init__(self):
        self.context_vars = {
            # built-in context vars
            "self": self,
            "os": os,
            "json": json,
            "yaml": yaml,
            "logger": logger,
            "groupby": groupby,
            "itemgetter": itemgetter,
        }

    def _process_variable(self, value: Any, context_vars: dict) -> Any:
        """Process a variable."""
        if isinstance(value, str):
            stripped_value = value.strip()
            if stripped_value.startswith('`') and stripped_value.endswith('`'):
                try:
                    # if the value is a python expression, evaluate it via python eval
                    return eval(stripped_value[1:-1], context_vars)
                except Exception as e:
                    logger.error(f"❌ [ConfigInterpreter] Error evaluating variable: [{type(e)}: {e}]\n{stripped_value} ")
                    raise e
            else:
                # if the value is a string, format it with the format
                return value.format(**context_vars)
        else:
            return value

    def _process_context_vars(self, context_config: dict, context_vars: dict, local_context_vars: Optional[dict] = None) -> dict:
        """Process context variables."""
        for key, value in context_config.items():
            # recursively process the context variables
            if isinstance(value, dict):
                if local_context_vars is None:
                    context_vars[key] = self._process_context_vars(value, context_vars, local_context_vars={})
                else:
                    local_context_vars[key] = self._process_context_vars(value, context_vars, local_context_vars={})
            else:
                if local_context_vars is None:
                    context_vars[key] = self._process_variable(value, context_vars)
                else:
                    local_context_vars[key] = self._process_variable(value, context_vars)
        if local_context_vars is None:
            context_vars['__context__'] = context_vars
            return context_vars
        else:
            return local_context_vars

    def _process_args_vars(self, args_config: dict, context_vars: dict) -> dict:
        """Process input variables."""
        result = {}
        for key, value in args_config.items():
            if isinstance(value, dict):
                result[key] = self._process_args_vars(value, context_vars)
            else:
                result[key] = self._process_variable(value, context_vars)
        return result

    async def _process_steps(self, runtime: Any, steps_config: dict, context_vars: dict):
        """Process steps."""
        try:
            # start processing the worker config, create a new context_vars dictionary
            # steps_context_vars = self.context_vars | context_vars
            # steps_context_vars['__runtime__'] = runtime
            # steps_context_vars['__context__'] = steps_context_vars

            # process the steps
            error_encountered = False
            for step_config in steps_config:
                try:
                    if 'type' not in step_config:
                        logger.error(f"🔴 [_process_steps] Step type not found in step: {step_config}")
                        error_encountered = True
                        break

                    step_type = step_config['type']
                    if step_type == 'endpoint':
                        success, extra_info = await self._process_endpoint(runtime, step_config, context_vars)
                        if extra_info is not None and 'wait_for' in extra_info:
                            wait_for_start_time = time.time()
                            wait_for = extra_info.get('wait_for', True)
                            interval = extra_info.get('interval', 15)
                            timeout = extra_info.get('timeout', 3600)
                            wait_for_config = extra_info.get('config', {})
                            while not wait_for and time.time() - wait_for_start_time < timeout:
                                logger.info(f"⏳ [_process_steps] Waiting for endpoint: {wait_for_config} [{wait_for}]")
                                await asyncio.sleep(interval)
                                success, extra_info = await self._process_endpoint(runtime, step_config, context_vars)
                                wait_for = extra_info.get('wait_for', True)
                                interval = extra_info.get('interval', 15)
                                timeout = extra_info.get('timeout', 3600)
                        if not success:
                            error_encountered = True
                            break
                    elif step_type == 'iterator':
                        success = await self._process_iterator(runtime, step_config, context_vars)
                        if not success:
                            error_encountered = True
                            break
                    else:
                        logger.error(f"🔴 [_process_steps] Step type not found in step: {step_config}")
                        error_encountered = True
                        break

                except Exception as e:
                    # assume each step depend on each other, always break the steps if current step fails
                    logger.error(f"🔴 [_process_steps] Error processing step: {step_config} [{type(e)}: {e}]")
                    logger.error(traceback.format_exc())
                    error_encountered = True
                    break

            if error_encountered:
                logger.error(f"🔴 [_process_steps] Error encountered in step: {step_config}")
                return None
            
            return True

        except Exception as e:
            logger.error(f"🔴 [_process_steps] error: [{type(e)}: {e}]")
            logger.error(traceback.format_exc())
            return None


    async def _process_endpoint(self,
        runtime: Any,
        endpoint_config: dict,
        context_vars: dict,
    ) -> tuple[bool, Optional[dict]]: # (success, wait_for_info)
        """Process an endpoint."""
        # check if the endpoint config is valid
        if endpoint_config.get('type', None) != "endpoint" or endpoint_config.get('endpoint', None) is None:
            logger.error(f"🔴 [_process_endpoint] Endpoint not found in endpoint config: {endpoint_config}")
            return False, None

        # start processing the worker config, create a new context_vars dictionary
        # endpoint_context_vars = self.context_vars | context_vars
        # endpoint_context_vars['__runtime__'] = runtime
        # endpoint_context_vars['__context__'] = endpoint_context_vars

        try:
            # get the endpoint instance and function
            endpoint = endpoint_config['endpoint'].split('.')
            endpoint_class_name = endpoint[0]
            endpoint_method_name = endpoint[1]
            # get self.{endpoint_class}
            endpoint_instance = getattr(runtime, endpoint_class_name)
            # get self.{endpoint_class}.{endpoint_method}
            endpoint_function = getattr(endpoint_instance, endpoint_method_name)

        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_endpoint] Error getting endpoint: {endpoint_config} [{type(e)}: {e}]")
            logger.error(traceback.format_exc())
            return False, None

        try:
            if 'context_vars' in endpoint_config:
                self._process_context_vars(endpoint_config['context_vars'], context_vars)

        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_endpoint] Error processing endpoint context variables: {endpoint_config['context_vars']} [{type(e)}: {e}]")
            return False, None

        try:
            # get the args
            args_config = endpoint_config.get('args', {})
            args_vars = self._process_args_vars(args_config, context_vars)

        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_endpoint] Error processing args: {endpoint_config} [{type(e)}: {e}]")
            return False, None

        try:
            # call the endpoint function
            record_time = True if 'returns' in endpoint_config else False
            if record_time:
                start_time = time.time()
            # check if the endpoint function is async
            if asyncio.iscoroutinefunction(endpoint_function):
                result = await endpoint_function(**args_vars)
            else:
                result = endpoint_function(**args_vars)
            if record_time:
                end_time = time.time()
                context_vars['__endpoint_time__'] = end_time - start_time

        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_endpoint] Error calling endpoint: {endpoint_config} [{type(e)}: {e}]")
            # log stack track only when actually calling the endpoint
            logger.error(traceback.format_exc())
            return False, None

        try:
            if 'returns' in endpoint_config:
                returns_config = endpoint_config['returns']
                context_vars['__result__'] = result
                # process error_if
                if 'error_if' in returns_config:
                    error_if = self._process_variable(returns_config['error_if'], context_vars)
                    if error_if:
                        logger.error(f"🔴 [_process_endpoint] Error in endpoint [{endpoint_class_name}.{endpoint_method_name}]: {result}")
                        return False, None

                # now we don't have any errors, process the return in context variables
                if 'context_vars' in returns_config:
                    self._process_context_vars(returns_config['context_vars'], context_vars)

                # process save_to
                if 'save_to' in returns_config:
                    for save_to_config in returns_config['save_to']:
                        path = self._process_variable(save_to_config['path'], context_vars)
                        data = self._process_variable(save_to_config['data'], context_vars)
                        format = save_to_config['format'] if 'format' in save_to_config else 'text'
                        runtime.recorder.save(path, data, format=format)

                # process logging
                if 'logging' in returns_config:
                    log_config = returns_config['logging']
                    try:
                        log_method = getattr(runtime, log_config.get('method', None))
                        if not log_method:
                            logger.warning(f"🔴 [_process_endpoint] Logging method not found: {log_config}")
                        # check if log_method is async
                        if asyncio.iscoroutinefunction(log_method):
                            await log_method(result, context_vars)
                        else:
                            log_method(result, context_vars)
                    except Exception as e:
                        logger.warning(f"🔴 [_process_endpoint] Error processing logging: {log_config} [{type(e)}: {e}]")

                # process wait_for
                if 'wait_for' in returns_config:
                    wait_for_config = returns_config['wait_for']
                    condition = self._process_variable(wait_for_config['condition'], context_vars)
                    interval = self._process_variable(wait_for_config['interval'], context_vars)
                    timeout = self._process_variable(wait_for_config['timeout'], context_vars)
                    return False, {
                        "wait_for": condition,
                        "interval": interval,
                        "timeout": timeout,
                        "config": wait_for_config,
                    }
            
        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_endpoint] Error processing returns: [endpoint_config={endpoint_config}] [{type(e)}: {e}]")
            logger.error(traceback.format_exc())
            return False, None

        return True, None


    async def _process_iterator(self, runtime: Any, iterator_config: dict, context_vars: dict) -> bool:
        """Process an iterator."""
        # check if the endpoint config is valid
        if iterator_config.get('type', None) != "iterator" or iterator_config.get('collection', None) is None:
            logger.error(f"🔴 [_process_iterator] Iterator not found in iterator config: {iterator_config}")
            return False

        # start processing the worker config, create a new context_vars dictionary
        # iterator_context_vars = self.context_vars | context_vars
        # iterator_context_vars['__runtime__'] = runtime
        # iterator_context_vars['__context__'] = iterator_context_vars

        try:
            collection = self._process_variable(iterator_config['collection'], context_vars)
        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [_process_iterator] Error processing collection: {iterator_config['collection']} [{type(e)}: {e}]")
            return False

        try:
            # if collection is a list, process the steps for each item in the list
            if isinstance(collection, list):
                for idx, item in enumerate(collection):
                    context_vars['__idx__'] = idx
                    context_vars['__item__'] = item
                    # add roe level context vars
                    try:
                        if 'context_vars' in iterator_config:
                            self._process_context_vars(iterator_config['context_vars'], context_vars)
                    except Exception as e:
                        # assume each step depend on each other, always break the steps if current step fails
                        logger.error(f"🔴 [_process_iterator] Error processing iterator context variables: {iterator_config['context_vars']} [{type(e)}: {e}]")
                        return False
                    # process the steps
                    success = await self._process_steps(runtime, iterator_config['steps'], context_vars)
                    if not success:
                        logger.error(f"🔴 [_process_iterator] Error processing [__idx__: {idx}]: [__item__: {item}]")
                        return False
            # if collection is a dict, process the steps for each key-value pair in the dict
            elif isinstance(collection, dict):
                for key, value in collection.items():
                    context_vars['__key__'] = key
                    context_vars['__item__'] = value                
                    # add roe level context vars
                    try:
                        if 'context_vars' in iterator_config:
                            self._process_context_vars(iterator_config['context_vars'], context_vars)
                    except Exception as e:
                        # assume each step depend on each other, always break the steps if current step fails
                        logger.error(f"🔴 [_process_iterator] Error processing iterator context variables: {iterator_config['context_vars']} [{type(e)}: {e}]")
                        return False
                    # process the steps
                    success = await self._process_steps(runtime, iterator_config['steps'], context_vars)
                    if not success:
                        logger.error(f"🔴 [_process_iterator] Error processing [__key__: {key}]: [__item__: {value}]")
                        return False
            elif isinstance(collection, Iterable):  
                for item in collection:
                    context_vars['__item__'] = item
                    # add roe level context vars
                    try:
                        if 'context_vars' in iterator_config:
                            self._process_context_vars(iterator_config['context_vars'], context_vars)
                    except Exception as e:
                        # assume each step depend on each other, always break the steps if current step fails
                        logger.error(f"🔴 [_process_iterator] Error processing iterator context variables: {iterator_config['context_vars']} [{type(e)}: {e}]")
                        return False
                    # process the steps
                    success = await self._process_steps(runtime, iterator_config['steps'], context_vars)
                    if not success:
                        logger.error(f"🔴 [_process_iterator] Error processing [__item__: {item}]")
                        return False
            else:
                logger.error(f"🔴 [_process_iterator] Iterator is not Iterable: [{type(collection)}] [{iterator_config['collection']}]")
                return False

        except Exception as e:
            logger.error(f"🔴 [_process_iterator] Error processing iterator: {iterator_config['iterator']} [{type(e)}: {e}]")
        
        return True

    async def execute(self, runtime: Any, config: dict, context_vars: Optional[dict] = None) -> bool:
        """Execute the config."""
        # start processing the worker config, create a new context_vars dictionary
        context_vars = (context_vars or {}) | self.context_vars
        context_vars['__runtime__'] = runtime
        context_vars['__context__'] = context_vars

        try:
            if 'context_vars' in config:
                self._process_context_vars(config['context_vars'], context_vars)

        except Exception as e:
            # assume each step depend on each other, always break the steps if current step fails
            logger.error(f"🔴 [EndpointInterpreter] Error processing context variables: {config['context_vars']} [{type(e)}: {e}]")
            return False

        try:
            return await self._process_steps(runtime, config['steps'], context_vars)
        except Exception as e:
            logger.error(f"🔴 [EndpointInterpreter] Error executing config: {config} [{type(e)}: {e}]")
            logger.error(traceback.format_exc())
            return False

    def prepare_context_vars(self, runtime: Any, context_config: dict, context_vars: Optional[dict] = None) -> dict:
        """Prepare context variables."""
        # start processing the worker config, create a new context_vars dictionary
        context_vars = (context_vars or {}) | self.context_vars
        context_vars['__runtime__'] = runtime
        context_vars['__context__'] = context_vars
        
        # process the context variables
        return self._process_context_vars(context_config, context_vars)
import asyncio
import argparse
import traceback
from logger import logger
from globalRegClient import GlobalRegClient
from trainerSFT import sft_train_block, sft_get_trainer
from trainerGRPO import grpo_train_block, grpo_get_trainer

client = GlobalRegClient()

async def main_loop(prefix_tag: str, test_mode: bool):

    queues = await client.get_queues()
    logger.info(f"🌀 [trainerMain] Main loop started for prefix: {prefix_tag}")

    # initialize trainers (for now, we only have sft and grpo)
    sft_trainer = sft_get_trainer(None, prefix_tag)
    grpo_trainer = grpo_get_trainer(sft_trainer, prefix_tag)

    callback = lambda checkpoint_path: logger.info(f"📞 [trainerMain] Callback: [{checkpoint_path}]")

    while True:
        try:
            sft_qsize = await client.qsize('trainer.sft')
            grpo_qsize = await client.qsize('trainer.grpo')

            if grpo_qsize['size'] > sft_qsize['size']:
                grpo_item = await client.dequeue(queue_name="trainer.grpo")
                if grpo_item['prefix_tag'] != prefix_tag:
                    logger.error(f"❌ [trainerMain] Skipping item with prefix: {grpo_item['prefix_tag']}")
                    continue
                if not test_mode and 'test_mode' in grpo_item and grpo_item['test_mode']:
                    logger.info(f"🔍 [trainerMain] Skipping test mode item with prefix: {grpo_item['prefix_tag']}")
                    continue
                logger.info(f"🧊 [trainerMain] Training GRPO block: {grpo_item['prefix_tag']}")
                grpo_train_block(
                    trainer=grpo_trainer,
                    prefix_tag=grpo_item['prefix_tag'],
                    epoch_id=grpo_item['epoch_id'],
                    block_id=grpo_item['block_id'],
                    input_tag=grpo_item['input_tag'],
                    input_dir=grpo_item['input_dir'],
                    callback=callback
                )
            elif sft_qsize['size'] > 0:
                sft_item = await client.dequeue(queue_name="trainer.sft")
                if sft_item['prefix_tag'] != prefix_tag:
                    logger.error(f"❌ [trainerMain] Skipping item with prefix: {sft_item['prefix_tag']}")
                    continue
                if not test_mode and 'test_mode' in sft_item and sft_item['test_mode']:
                    logger.info(f"🔍 [trainerMain] Skipping test mode item with prefix: {sft_item['prefix_tag']}")
                    continue
                logger.info(f"🧊 [trainerMain] Training SFT block: {sft_item['prefix_tag']}")
                sft_train_block(
                    trainer=sft_trainer,
                    prefix_tag=sft_item['prefix_tag'],
                    epoch_id=sft_item['epoch_id'],
                    block_id=sft_item['block_id'],
                    input_tag=sft_item['input_tag'],
                    input_dir=sft_item['input_dir'],
                    callback=callback
                )
            else:
                logger.info(f"🔍 [trainerMain] No items to process, sleeping for [10] seconds")
                await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"❌ [trainerMain] Error: {e}")
            traceback.print_exc()
            await asyncio.sleep(5)

async def main():
    parser = argparse.ArgumentParser(description="Train a model using mixed SFT and GRPO trainers")
    parser.add_argument("--prefix_tag", type=str, default="KC_0.1.0")
    parser.add_argument("--test_mode", action="store_true")
    args = parser.parse_args()

    if args.test_mode:
        await client.enqueue('trainer.sft', {
            'prefix_tag': args.prefix_tag,
            'epoch_id': 0,
            'block_id': 0,
            'input_tag': 'v0.1_20250725_020900',
            'input_dir': '~/.critique',
            'test_mode': args.test_mode
        })
        await client.enqueue('trainer.grpo', {
            'prefix_tag': args.prefix_tag,
            'epoch_id': 0,
            'block_id': 0,
            'input_tag': 'v0.1_20250725_020900',
            'input_dir': '~/.codeGenEval',
            'test_mode': args.test_mode
        })

    # run the main loop
    await main_loop(args.prefix_tag, args.test_mode)

if __name__ == "__main__":
    # test
    asyncio.run(main())

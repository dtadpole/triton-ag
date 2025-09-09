test sync save and load

————— SAVE —————

2025-08-31 14:10:06.364 | [2101253] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:10:06.367 | [2101251] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:10:06.369 | [2101254] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:10:06.377 | [2101252] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict

2025-08-31 14:10:33.944 | [2101251] | INFO     | __main__:_save_checkpoint:514 | 📊 [EngineFSDP-0] Before save, self model state hash: [e90d0da5775fdfa11cceef93bd9fddae]
2025-08-31 14:10:33.944 | [2101251] | INFO     | __main__:_save_checkpoint:515 | 📊 [EngineFSDP-0] Before save, self optimizer state hash: [9b3fd0433b2a193f548a94b3383c6590]
2025-08-31 14:10:33.944 | [2101251] | INFO     | __main__:_save_checkpoint:516 | 📊 [EngineFSDP-0] Before save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:10:33.372 | [2101252] | INFO     | __main__:_save_checkpoint:514 | 📊 [EngineFSDP-1] Before save, self model state hash: [c6be73f515e4d3b02a04cd78454ad121]
2025-08-31 14:10:33.372 | [2101252] | INFO     | __main__:_save_checkpoint:515 | 📊 [EngineFSDP-1] Before save, self optimizer state hash: [28b76e2666400a6f147d581d49d58b40]
2025-08-31 14:10:33.372 | [2101252] | INFO     | __main__:_save_checkpoint:516 | 📊 [EngineFSDP-1] Before save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:10:33.613 | [2101253] | INFO     | __main__:_save_checkpoint:514 | 📊 [EngineFSDP-2] Before save, self model state hash: [78c8a24bb48d329246c7df84c7f8c6e0]
2025-08-31 14:10:33.613 | [2101253] | INFO     | __main__:_save_checkpoint:515 | 📊 [EngineFSDP-2] Before save, self optimizer state hash: [a354653738d99f587d39333b653d7d1a]
2025-08-31 14:10:33.613 | [2101253] | INFO     | __main__:_save_checkpoint:516 | 📊 [EngineFSDP-2] Before save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:10:34.578 | [2101254] | INFO     | __main__:_save_checkpoint:514 | 📊 [EngineFSDP-3] Before save, self model state hash: [0bc8696077a9df32b4b8574956e621cb]
2025-08-31 14:10:34.579 | [2101254] | INFO     | __main__:_save_checkpoint:515 | 📊 [EngineFSDP-3] Before save, self optimizer state hash: [d17f82c5eaf761db85df476bc193f49d]
2025-08-31 14:10:34.579 | [2101254] | INFO     | __main__:_save_checkpoint:516 | 📊 [EngineFSDP-3] Before save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]



2025-08-31 14:11:12.930 | [2101252] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:11:12.930 | [2101253] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:11:12.930 | [2101254] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:11:12.931 | [2101251] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict

2025-08-31 14:11:40.225 | [2101251] | INFO     | __main__:_save_checkpoint:544 | 📊 [EngineFSDP-0] After save, self model state hash: [e90d0da5775fdfa11cceef93bd9fddae]
2025-08-31 14:11:40.226 | [2101251] | INFO     | __main__:_save_checkpoint:545 | 📊 [EngineFSDP-0] After save, self optimizer state hash: [9b3fd0433b2a193f548a94b3383c6590]
2025-08-31 14:11:40.226 | [2101251] | INFO     | __main__:_save_checkpoint:546 | 📊 [EngineFSDP-0] After save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:11:40.565 | [2101252] | INFO     | __main__:_save_checkpoint:544 | 📊 [EngineFSDP-1] After save, self model state hash: [c6be73f515e4d3b02a04cd78454ad121]
2025-08-31 14:11:40.565 | [2101252] | INFO     | __main__:_save_checkpoint:545 | 📊 [EngineFSDP-1] After save, self optimizer state hash: [28b76e2666400a6f147d581d49d58b40]
2025-08-31 14:11:40.565 | [2101252] | INFO     | __main__:_save_checkpoint:546 | 📊 [EngineFSDP-1] After save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:11:39.599 | [2101253] | INFO     | __main__:_save_checkpoint:544 | 📊 [EngineFSDP-2] After save, self model state hash: [78c8a24bb48d329246c7df84c7f8c6e0]
2025-08-31 14:11:39.599 | [2101253] | INFO     | __main__:_save_checkpoint:545 | 📊 [EngineFSDP-2] After save, self optimizer state hash: [a354653738d99f587d39333b653d7d1a]
2025-08-31 14:11:39.599 | [2101253] | INFO     | __main__:_save_checkpoint:546 | 📊 [EngineFSDP-2] After save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:11:40.638 | [2101254] | INFO     | __main__:_save_checkpoint:544 | 📊 [EngineFSDP-3] After save, self model state hash: [0bc8696077a9df32b4b8574956e621cb]
2025-08-31 14:11:40.638 | [2101254] | INFO     | __main__:_save_checkpoint:545 | 📊 [EngineFSDP-3] After save, self optimizer state hash: [d17f82c5eaf761db85df476bc193f49d]
2025-08-31 14:11:40.638 | [2101254] | INFO     | __main__:_save_checkpoint:546 | 📊 [EngineFSDP-3] After save, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]


————— LOAD —————

2025-08-31 14:14:44.619 | [2153466] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:14:44.619 | [2153459] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:14:44.620 | [2153464] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:14:44.626 | [2153462] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict

2025-08-31 14:15:05.897 | [2153459] | INFO     | __main__:_load_checkpoint:446 | 📊 [EngineFSDP-0] Before load, self model state hash: [df0fb4b573ddf7f5f66d922a98ed1c00]
2025-08-31 14:15:05.898 | [2153459] | INFO     | __main__:_load_checkpoint:447 | 📊 [EngineFSDP-0] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 14:15:05.898 | [2153459] | INFO     | __main__:_load_checkpoint:448 | 📊 [EngineFSDP-0] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 14:15:05.382 | [2153462] | INFO     | __main__:_load_checkpoint:446 | 📊 [EngineFSDP-1] Before load, self model state hash: [ff1ec77a2922ad02aba8384c99b1ac1f]
2025-08-31 14:15:05.382 | [2153462] | INFO     | __main__:_load_checkpoint:447 | 📊 [EngineFSDP-1] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 14:15:05.382 | [2153462] | INFO     | __main__:_load_checkpoint:448 | 📊 [EngineFSDP-1] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 14:15:05.153 | [2153464] | INFO     | __main__:_load_checkpoint:446 | 📊 [EngineFSDP-2] Before load, self model state hash: [3910074db59bd113d150a287a16446d8]
2025-08-31 14:15:05.153 | [2153464] | INFO     | __main__:_load_checkpoint:447 | 📊 [EngineFSDP-2] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 14:15:05.153 | [2153464] | INFO     | __main__:_load_checkpoint:448 | 📊 [EngineFSDP-2] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 14:15:05.098 | [2153466] | INFO     | __main__:_load_checkpoint:446 | 📊 [EngineFSDP-3] Before load, self model state hash: [8cca0baa0dfac2ef77d45346f06e4956]
2025-08-31 14:15:05.098 | [2153466] | INFO     | __main__:_load_checkpoint:447 | 📊 [EngineFSDP-3] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 14:15:05.098 | [2153466] | INFO     | __main__:_load_checkpoint:448 | 📊 [EngineFSDP-3] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]



2025-08-31 14:15:32.485 | [2153466] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:32.485 | [2153459] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:32.486 | [2153462] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:32.487 | [2153464] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict

2025-08-31 14:15:53.157 | [2153459] | INFO     | __main__:_load_checkpoint:470 | 📊 [EngineFSDP-0] Loaded app model state hash: [e90d0da5775fdfa11cceef93bd9fddae]
2025-08-31 14:15:53.157 | [2153459] | INFO     | __main__:_load_checkpoint:471 | 📊 [EngineFSDP-0] Loaded App optimizer state hash: [9b3fd0433b2a193f548a94b3383c6590]
2025-08-31 14:15:53.157 | [2153459] | INFO     | __main__:_load_checkpoint:472 | 📊 [EngineFSDP-0] Loaded App scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:15:53.470 | [2153462] | INFO     | __main__:_load_checkpoint:470 | 📊 [EngineFSDP-1] Loaded app model state hash: [c6be73f515e4d3b02a04cd78454ad121]
2025-08-31 14:15:53.471 | [2153462] | INFO     | __main__:_load_checkpoint:471 | 📊 [EngineFSDP-1] Loaded App optimizer state hash: [28b76e2666400a6f147d581d49d58b40]
2025-08-31 14:15:53.471 | [2153462] | INFO     | __main__:_load_checkpoint:472 | 📊 [EngineFSDP-1] Loaded App scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:15:52.346 | [2153464] | INFO     | __main__:_load_checkpoint:470 | 📊 [EngineFSDP-2] Loaded app model state hash: [78c8a24bb48d329246c7df84c7f8c6e0]
2025-08-31 14:15:52.346 | [2153464] | INFO     | __main__:_load_checkpoint:471 | 📊 [EngineFSDP-2] Loaded App optimizer state hash: [a354653738d99f587d39333b653d7d1a]
2025-08-31 14:15:52.346 | [2153464] | INFO     | __main__:_load_checkpoint:472 | 📊 [EngineFSDP-2] Loaded App scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:15:53.433 | [2153466] | INFO     | __main__:_load_checkpoint:470 | 📊 [EngineFSDP-3] Loaded app model state hash: [0bc8696077a9df32b4b8574956e621cb]
2025-08-31 14:15:53.433 | [2153466] | INFO     | __main__:_load_checkpoint:471 | 📊 [EngineFSDP-3] Loaded App optimizer state hash: [d17f82c5eaf761db85df476bc193f49d]
2025-08-31 14:15:53.434 | [2153466] | INFO     | __main__:_load_checkpoint:472 | 📊 [EngineFSDP-3] Loaded App scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]


2025-08-31 14:15:55.269 | [2153462] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:55.270 | [2153466] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:55.270 | [2153464] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 14:15:55.278 | [2153459] | INFO     | __main__:get_model_state_hash:849 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict

2025-08-31 14:16:18.788 | [2153459] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-0] After load, self model state hash: [e90d0da5775fdfa11cceef93bd9fddae]
2025-08-31 14:16:18.789 | [2153459] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-0] After load, self optimizer state hash: [9b3fd0433b2a193f548a94b3383c6590]
2025-08-31 14:16:18.789 | [2153459] | INFO     | __main__:_load_checkpoint:480 | 📊 [EngineFSDP-0] After load, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:16:18.343 | [2153462] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-1] After load, self model state hash: [c6be73f515e4d3b02a04cd78454ad121]
2025-08-31 14:16:18.343 | [2153462] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-1] After load, self optimizer state hash: [28b76e2666400a6f147d581d49d58b40]
2025-08-31 14:16:18.343 | [2153462] | INFO     | __main__:_load_checkpoint:480 | 📊 [EngineFSDP-1] After load, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:16:21.274 | [2153464] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-2] After load, self model state hash: [78c8a24bb48d329246c7df84c7f8c6e0]
2025-08-31 14:16:21.274 | [2153464] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-2] After load, self optimizer state hash: [a354653738d99f587d39333b653d7d1a]
2025-08-31 14:16:21.274 | [2153464] | INFO     | __main__:_load_checkpoint:480 | 📊 [EngineFSDP-2] After load, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]

2025-08-31 14:16:21.016 | [2153466] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-3] After load, self model state hash: [0bc8696077a9df32b4b8574956e621cb]
2025-08-31 14:16:21.016 | [2153466] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-3] After load, self optimizer state hash: [d17f82c5eaf761db85df476bc193f49d]
2025-08-31 14:16:21.016 | [2153466] | INFO     | __main__:_load_checkpoint:480 | 📊 [EngineFSDP-3] After load, self scheduler state hash: [1047eeec986ae75dc9627d1b862d2fd9]



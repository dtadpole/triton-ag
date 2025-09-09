Async SAVE and LOAD

————— SAVE —————

2025-08-31 16:49:24.410 | [3235234] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:49:24.410 | [3235236] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:49:24.412 | [3235235] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:49:24.412 | [3235237] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict

2025-08-31 16:49:47.188 | [3235234] | INFO     | __main__:_save_checkpoint:527 | 📊 [EngineFSDP-0] Before save, self model state hash: [353e1bb0b2e1f021c70561660a22a20d]
2025-08-31 16:49:47.188 | [3235234] | INFO     | __main__:_save_checkpoint:528 | 📊 [EngineFSDP-0] Before save, self optimizer state hash: [03636edaf4a52a2665bd40c4c8bb07c1]
2025-08-31 16:49:47.188 | [3235234] | INFO     | __main__:_save_checkpoint:529 | 📊 [EngineFSDP-0] Before save, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:49:51.218 | [3235235] | INFO     | __main__:_save_checkpoint:527 | 📊 [EngineFSDP-1] Before save, self model state hash: [41be8beee8121cc61468556731fd2c26]
2025-08-31 16:49:51.218 | [3235235] | INFO     | __main__:_save_checkpoint:528 | 📊 [EngineFSDP-1] Before save, self optimizer state hash: [1ae46272f8b509d8034aed5a4d40654c]
2025-08-31 16:49:51.218 | [3235235] | INFO     | __main__:_save_checkpoint:529 | 📊 [EngineFSDP-1] Before save, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:49:45.640 | [3235236] | INFO     | __main__:_save_checkpoint:527 | 📊 [EngineFSDP-2] Before save, self model state hash: [09a8f9a515788b910ab6db0206f7b722]
2025-08-31 16:49:45.640 | [3235236] | INFO     | __main__:_save_checkpoint:528 | 📊 [EngineFSDP-2] Before save, self optimizer state hash: [0b3fcd119d69ee469478dec747d4738a]
2025-08-31 16:49:45.640 | [3235236] | INFO     | __main__:_save_checkpoint:529 | 📊 [EngineFSDP-2] Before save, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:49:51.570 | [3235237] | INFO     | __main__:_save_checkpoint:527 | 📊 [EngineFSDP-3] Before save, self model state hash: [ca329b1e6ae09ee8ede4ff9b28f6d1e3]
2025-08-31 16:49:51.570 | [3235237] | INFO     | __main__:_save_checkpoint:528 | 📊 [EngineFSDP-3] Before save, self optimizer state hash: [4c2978eaf123043cc5a273582483a114]
2025-08-31 16:49:51.570 | [3235237] | INFO     | __main__:_save_checkpoint:529 | 📊 [EngineFSDP-3] Before save, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]


————— LOAD —————

2025-08-31 16:53:25.398 | [3328328] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:53:25.399 | [3328327] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:53:25.399 | [3328326] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:53:25.409 | [3328325] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict

2025-08-31 16:53:46.598 | [3328325] | INFO     | __main__:_load_checkpoint:452 | 📊 [EngineFSDP-0] Before load, self model state hash: [df0fb4b573ddf7f5f66d922a98ed1c00]
2025-08-31 16:53:46.598 | [3328325] | INFO     | __main__:_load_checkpoint:453 | 📊 [EngineFSDP-0] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 16:53:46.598 | [3328325] | INFO     | __main__:_load_checkpoint:454 | 📊 [EngineFSDP-0] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 16:53:46.259 | [3328326] | INFO     | __main__:_load_checkpoint:452 | 📊 [EngineFSDP-1] Before load, self model state hash: [ff1ec77a2922ad02aba8384c99b1ac1f]
2025-08-31 16:53:46.259 | [3328326] | INFO     | __main__:_load_checkpoint:453 | 📊 [EngineFSDP-1] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 16:53:46.259 | [3328326] | INFO     | __main__:_load_checkpoint:454 | 📊 [EngineFSDP-1] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 16:53:44.944 | [3328327] | INFO     | __main__:_load_checkpoint:452 | 📊 [EngineFSDP-2] Before load, self model state hash: [3910074db59bd113d150a287a16446d8]
2025-08-31 16:53:44.944 | [3328327] | INFO     | __main__:_load_checkpoint:453 | 📊 [EngineFSDP-2] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 16:53:44.944 | [3328327] | INFO     | __main__:_load_checkpoint:454 | 📊 [EngineFSDP-2] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]

2025-08-31 16:53:53.097 | [3328328] | INFO     | __main__:_load_checkpoint:452 | 📊 [EngineFSDP-3] Before load, self model state hash: [8cca0baa0dfac2ef77d45346f06e4956]
2025-08-31 16:53:53.098 | [3328328] | INFO     | __main__:_load_checkpoint:453 | 📊 [EngineFSDP-3] Before load, self optimizer state hash: [10b7c2c94575f497215616478d3d3861]
2025-08-31 16:53:53.098 | [3328328] | INFO     | __main__:_load_checkpoint:454 | 📊 [EngineFSDP-3] Before load, self scheduler state hash: [3324a7f82c52b1f649cc3595fd4ea660]




2025-08-31 16:54:23.327 | [3328327] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:23.327 | [3328325] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:23.331 | [3328326] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:23.335 | [3328328] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict

2025-08-31 16:54:45.466 | [3328325] | INFO     | __main__:_load_checkpoint:477 | 📊 [EngineFSDP-0] Loaded app model state hash: [353e1bb0b2e1f021c70561660a22a20d]
2025-08-31 16:54:45.466 | [3328325] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-0] Loaded App optimizer state hash: [03636edaf4a52a2665bd40c4c8bb07c1]
2025-08-31 16:54:45.466 | [3328325] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-0] Loaded App scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:54:43.018 | [3328326] | INFO     | __main__:_load_checkpoint:477 | 📊 [EngineFSDP-1] Loaded app model state hash: [41be8beee8121cc61468556731fd2c26]
2025-08-31 16:54:43.019 | [3328326] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-1] Loaded App optimizer state hash: [1ae46272f8b509d8034aed5a4d40654c]
2025-08-31 16:54:43.019 | [3328326] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-1] Loaded App scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:54:43.488 | [3328327] | INFO     | __main__:_load_checkpoint:477 | 📊 [EngineFSDP-2] Loaded app model state hash: [09a8f9a515788b910ab6db0206f7b722]
2025-08-31 16:54:43.488 | [3328327] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-2] Loaded App optimizer state hash: [0b3fcd119d69ee469478dec747d4738a]
2025-08-31 16:54:43.488 | [3328327] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-2] Loaded App scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:54:50.611 | [3328328] | INFO     | __main__:_load_checkpoint:477 | 📊 [EngineFSDP-3] Loaded app model state hash: [ca329b1e6ae09ee8ede4ff9b28f6d1e3]
2025-08-31 16:54:50.612 | [3328328] | INFO     | __main__:_load_checkpoint:478 | 📊 [EngineFSDP-3] Loaded App optimizer state hash: [4c2978eaf123043cc5a273582483a114]
2025-08-31 16:54:50.612 | [3328328] | INFO     | __main__:_load_checkpoint:479 | 📊 [EngineFSDP-3] Loaded App scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]




2025-08-31 16:54:52.405 | [3328325] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 0: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:52.406 | [3328327] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 2: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:52.407 | [3328326] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 1: Model has [2,047,683,840] parameters in state dict
2025-08-31 16:54:52.414 | [3328328] | INFO     | __main__:get_model_state_hash:864 | 🔍 Rank 3: Model has [2,047,683,840] parameters in state dict

2025-08-31 16:55:19.061 | [3328325] | INFO     | __main__:_load_checkpoint:485 | 📊 [EngineFSDP-0] After load, self model state hash: [353e1bb0b2e1f021c70561660a22a20d]
2025-08-31 16:55:19.062 | [3328325] | INFO     | __main__:_load_checkpoint:486 | 📊 [EngineFSDP-0] After load, self optimizer state hash: [03636edaf4a52a2665bd40c4c8bb07c1]
2025-08-31 16:55:19.062 | [3328325] | INFO     | __main__:_load_checkpoint:487 | 📊 [EngineFSDP-0] After load, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:55:17.544 | [3328326] | INFO     | __main__:_load_checkpoint:485 | 📊 [EngineFSDP-1] After load, self model state hash: [41be8beee8121cc61468556731fd2c26]
2025-08-31 16:55:17.545 | [3328326] | INFO     | __main__:_load_checkpoint:486 | 📊 [EngineFSDP-1] After load, self optimizer state hash: [1ae46272f8b509d8034aed5a4d40654c]
2025-08-31 16:55:17.545 | [3328326] | INFO     | __main__:_load_checkpoint:487 | 📊 [EngineFSDP-1] After load, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:55:17.944 | [3328327] | INFO     | __main__:_load_checkpoint:485 | 📊 [EngineFSDP-2] After load, self model state hash: [09a8f9a515788b910ab6db0206f7b722]
2025-08-31 16:55:17.944 | [3328327] | INFO     | __main__:_load_checkpoint:486 | 📊 [EngineFSDP-2] After load, self optimizer state hash: [0b3fcd119d69ee469478dec747d4738a]
2025-08-31 16:55:17.944 | [3328327] | INFO     | __main__:_load_checkpoint:487 | 📊 [EngineFSDP-2] After load, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]

2025-08-31 16:55:19.904 | [3328328] | INFO     | __main__:_load_checkpoint:485 | 📊 [EngineFSDP-3] After load, self model state hash: [ca329b1e6ae09ee8ede4ff9b28f6d1e3]
2025-08-31 16:55:19.905 | [3328328] | INFO     | __main__:_load_checkpoint:486 | 📊 [EngineFSDP-3] After load, self optimizer state hash: [4c2978eaf123043cc5a273582483a114]
2025-08-31 16:55:19.905 | [3328328] | INFO     | __main__:_load_checkpoint:487 | 📊 [EngineFSDP-3] After load, self scheduler state hash: [d7d3b5e60c6fb9af2c815cc8eb9b4194]


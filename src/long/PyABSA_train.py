# Train a model of aspect term extraction

import random

from pyabsa import AspectTermExtraction as ATEPC

config = ATEPC.ATEPCConfigManager.get_atepc_config_english()
config.model = ATEPC.ATEPCModelList.FAST_LCF_ATEPC
config.evaluate_begin = 0
config.max_seq_len = 512
config.pretrained_bert = 'yangheng/deberta-v3-base-absa'
config.l2reg = 1e-8
config.seed = random.randint(1, 100)
config.use_bert_spc = True
config.use_amp = False
config.cache_dataset = False

English_sets = ATEPC.ATEPCDatasetList.English_SemEval2016Task5

aspect_extractor = ATEPC.ATEPCTrainer(config=config,
                                      dataset=English_sets,
                                      checkpoint_save_mode=1,
                                      auto_device=True
                                      ).load_trained_model()

atepc_examples = ['But the staff was so nice to us .',
                  'But the staff was so horrible to us .',
                  r'Not only was the food outstanding , but the little ` perks \' were great .',
                  ]
aspect_extractor.batch_predict(inference_source=atepc_examples,
                               save_result=True,
                               # print the result
                               print_result=True,
                               # Predict the sentiment of extracted aspect terms
                               pred_sentiment=True,
                               )

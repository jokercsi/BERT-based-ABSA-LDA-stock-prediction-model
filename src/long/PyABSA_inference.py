# Inference Example of aspect term extraction

from pyabsa import AspectTermExtraction as ATEPC

# checkpoint_map = available_checkpoints(from_local=False)


aspect_extractor = ATEPC.AspectExtractor('multilingual',
                                         auto_device=True,  # False means load model on CPU
                                         cal_perplexity=True,
                                         )

inference_source = ATEPC.ATEPCDatasetList.SemEval
atepc_result = aspect_extractor.batch_predict(inference_source=inference_source,  #
                                              save_result=True,
                                              print_result=True,  # print the result
                                              pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                              )

print(atepc_result)
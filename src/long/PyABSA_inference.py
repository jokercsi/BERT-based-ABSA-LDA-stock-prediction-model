from pyabsa import ATEPCCheckpointManager

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint='english',
                                   auto_device=True  # False means load model on CPU
                                   )

# You can inference from a list of setences or a DatasetItem from PyABSA 
examples = ['Twitter CFO: We have a lot of work to do', ' but everybody in the world can benefit from Twitter']
inference_source = examples
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,  #
                          pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                          )

print(atepc_result)
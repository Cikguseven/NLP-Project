from detoxify import Detoxify
detoxify_model = Detoxify('unbiased', device='cuda')
print(detoxify_model.predict(['a', 'fuck', 'god damn it']))
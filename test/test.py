


from mderank import MDERank, MDERankConfig

cfg = MDERankConfig(lang="es",model_name_or_path="/Users/pablo/Downloads/maria/roberta-base-bne",model_type='roberta', log_level="INFO")
extractor = MDERank(cfg)

result = extractor.evaluate("../data/example/docsutf8","../data/example/keys",15)
print(result)
print(len(result[0]))


"""
result = extractor.extract("data/example/docsutf8",15)
print(result)
print(len(result[0]))
"""
print("fin")


# mderank

Library for MDERank model to AKE

It has been adapted and improved to work also in Spanish language. 

Original Paper: MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction
Original Repo: https://github.com/LinhanZ/mderank






## Install

This project has been developed under Python 3.9.6

Use requirements.txt and requirements-torch.txt to install the required libraries. You can use torch with and without gpu

Download also the required spacy models of Spanish/English

```
python -m spacy download en_core_web_sm
python -m spacy download es_core_news_sm

```



## Run 

If you want to evaluate a dataset as in the State of the Art, put the dataset in a data folder (data/datasetname/docsutf8 and data/datasetname/keys).
Configure parameters and arguments of evaluate.sh and run it. 

```
 bash eval.sh 
```
If you want to execute it over texts on a folder, configure and execute run.sh file
```
 bash run.sh 
```

## Docker run 
For a fast run use the dockerfile and this two commands. In these commands, mderank will read a folder named example with all the documents that are inside and it will create a file .key for each file with the keywords detected

```
docker build -t mderanklib .

``` 

```
docker run --rm -v ./example:/app/example mderanklib --dataset_dir example --batch_size 1  --doc_embed_mode max --log_dir log_path --model_name_or_path PlanTL-GOB-ES/roberta-base-bne --model_type roberta --dataset_name example --type_execution eval --k_value 15 --layer_num -1 --lang es --no_cuda
```


## Acknowledgments 

Este código se ha mejorado y adaptado en el marco del proyecto TeresIA, proyecto de investigación financiado con fondos de la Unión Europea Next GenerationEU / PRTR a través del Ministerio de Asuntos Económicos y Transformación Digital (hoy Ministerio para la Transformación Digital y de la Función Pública).







# InternshipMeduzzen
I have done some research into given [datasets](https://drive.google.com/drive/u/1/folders/16LkwOXX8XxDV45VgXaDXAQNJASUF71bS) and here is my opinion on an Email Classification dataset.
## Libraries I am going to use ##
- **spaCy**<br/>
  Modern NLP library, which supports around 70 languages and has pipelines for 25+ languages. As for now, I think I will use spaCy as a main NLP library in the project because it has built-in lemmatization and is more convenient for   
  smaller projects.
- **scikit-learn**<br/>
  It contains plenty of metrics and models, which might be used in the project (*TfIdfClassifier, confusion_matrix, classification_report*).
- **NLTK**<br/>
  This library might be useful, as it is more flexible than spaCy and gives more control of NLP pipeline.
- **Pandas**<br/>
  This library is useful for data structurization. For instance, I could structurize emails by *Rolle* mentioned in *ROLLEN_VORGANG* spreadsheet from Email Classification dataset.
- **TensorFlow Keras**<br/>
  TensorFlow offers a wide variety of pre-trained models for NLP, such as BERT, GPT or Word2Vec, which might be useful.
## Vision of the project ##
As I have done some research into E-Plus emails dataset, I believe that I will need to do these steps:
1. **Data extraction**: I will need to extract the data from text files given in the dataset.
2. **Data structurization**: Extracted data will be organized in a pandas DataFrame for further processing.
3. **Data pre-processing**: Text vectorivation via sklearn CountVectorizer.
4. **Training a model**: Training TfIdfTransformer model on a vectorized data.
5. **Making predictions & estimating results**: Making a dataset with predictions. Making confusion matrix and classification report for each email class.

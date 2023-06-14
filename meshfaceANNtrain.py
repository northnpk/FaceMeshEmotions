import preprocessFERplus as preprocess
import facemeshANN as classifier

data = preprocess.FERdata('challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
# result = data.get_df(mode='ANN', sample=True, sample_size=35)
result = data.get_df(mode='ANN')

train_df = result[result['usage'] == 'train'].drop(columns='usage').sample(frac=1, ignore_index=True)
val_df = result[result['usage'] == 'val'].drop(columns='usage').sample(frac=1, ignore_index=True)
test_df = result[result['usage'] == 'test'].drop(columns='usage').sample(frac=1, ignore_index=True)

model = classifier.ANNClassifier(input_size=478*3, output_size=8, dropout=0.3)
# model = classifier.ANNClassifier(input_size=48*48*3, output_size=8, dropout=0.3)
print(model)

model, test_loss, correct = classifier.trainmodel(model, train_df, val_df, test_df, epochs=100, lr=3e-4, batch_size=32)



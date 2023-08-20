import utils.preprocessFERplus as preprocess
import utils.imgViT as classifier

data = preprocess.FERdata('challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')
# data = preprocess.FERdata('facemesh_df.csv')
result = data.get_df(mode='IMG', sample=True, sample_size=100)
result = data.balance_df('up')
# data.save_df('./facemesh_df.csv')

print(result.head(10))

train_df = result[result['usage'] == 'train'].drop(columns='usage').sample(frac=1, ignore_index=True)
val_df = result[result['usage'] == 'val'].drop(columns='usage').sample(frac=1, ignore_index=True)
test_df = result[result['usage'] == 'test'].drop(columns='usage').sample(frac=1, ignore_index=True)

model = classifier.ViTClassifier(input_size=48, output_size=7, dropout=0.1)

print(model)
model, test_loss, correct = classifier.trainmodel(model, train_df, val_df, test_df, epochs=10, lr=3e-4, batch_size=32, plot=True, class_name = data.class_name[:-1])

# classifier.savemodel(model, save_path='./model/FERplusmeshANNRotate.pt')
import preprocessFERplus as preprocess
import facemeshGCN as classifier

data = preprocess.FERdata('challenges-in-representation-learning-facial-expression-recognition-challenge/icml_face_data.csv')

result = data.get_df(mode='GNN',sample=True, sample_size=100)
result = data.balance_df('up')
# data.save_df('./facemesh_df.csv')

print(result.head(10))

train_df = result[result['usage'] == 'train'].drop(columns='usage').sample(frac=1, ignore_index=True)
val_df = result[result['usage'] == 'val'].drop(columns='usage').sample(frac=1, ignore_index=True)
test_df = result[result['usage'] == 'test'].drop(columns='usage').sample(frac=1, ignore_index=True)

model = classifier.GCNClassifier(input_size=3, output_size=7, dropout=0.5)
# model = classifier.ANNClassifier(input_size=48*48*3, output_size=7, dropout=0.5)
# model = classifier.getmodel(model, './model/FERplusmeshANNColab.pt')
print(model)
model, test_loss, correct = classifier.trainmodel(model, train_df, val_df, test_df, epochs=100, lr=1e-7, batch_size=1024, plot=True, class_name = data.class_name)

# best_model = None
# min_loss = 99999.9
# for i in range(10):
#     model = classifier.ANNClassifier(input_size=478*3, output_size=7, dropout=0.5)
#     print(f'Model {i+1} start training')
#     model, test_loss, correct = classifier.trainmodel(model, train_df, val_df, test_df, epochs=100, lr=3e-4, batch_size=128)
#     if test_loss < min_loss:
#         best_model = model
#         min_loss = test_loss

# print(f'min loss is {np.array(min_loss).argmin()}')
# classifier.savemodel(model, save_path='./model/FERplusmeshANNBest.pt')

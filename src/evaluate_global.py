import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

CECRtrunc = {"A1": "A1", "A2": "A2", "B1": "B1", "B2": "B2", "C1": "B2", "C2": "B2"}
# Ordinal mapping
cefr_order = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3}



predictions_df = pd.read_csv('../predictions/predictions_global_mistral_large_zero.csv', sep='\t', index_col='text_indice')
print(predictions_df.columns)

#predictions_df['prediction_filtered'] = predictions_df['prediction'].str.extract(r'Niveau CECR\s*:\s*\*\*(.*?)\*\*')

predictions_df['prediction_filtered'] = predictions_df['prediction'].str.extract(r'\*\*(A1|A2|B1|B2|C1|C2)\*\*')


print(predictions_df["prediction_filtered"].value_counts())

predictions_df['prediction_truc'] = predictions_df['prediction_filtered'].map(CECRtrunc)

print(predictions_df["prediction_truc"].value_counts())




cm = confusion_matrix(y_true=predictions_df['classe'], y_pred=predictions_df['prediction_truc'])

labels = sorted(predictions_df['classe'].unique())
# Compute confusion matrix
cm = confusion_matrix(
    y_true=predictions_df['classe'],
    y_pred=predictions_df['prediction_truc'],
    labels=labels
)
# Display as DataFrame with labels as index and columns
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Accuracy (overall)
accuracy = accuracy_score(predictions_df['classe'], predictions_df['prediction_truc'])

print(predictions_df['classe'])
# Macro F1 score
macro_f1 = f1_score(predictions_df['classe'], predictions_df['prediction_truc'], average='macro')

# Map to integers
y_true = predictions_df['classe'].map(cefr_order)
y_pred = predictions_df['prediction_truc'].map(cefr_order)

# Adjacent accuracy: count if prediction is within ±1
adjacent_acc = ((y_true - y_pred).abs() <= 1).mean()

print("confusion matrix \n", cm_df)

print("accuracy ", accuracy)
print("macro_f1 ", macro_f1)

print("adjacent_acc ", adjacent_acc)

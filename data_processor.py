import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier


from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Ana klasör yolu
base_folder = "data\\raw_data"

# Boş bir DataFrame
all_data = pd.DataFrame()

# Alt klasörlerdeki CSV dosyalarının işlenmesi
for root, dirs, files in os.walk(base_folder):
    folder_name = os.path.basename(root)  # Alt klasör adı, sınıf etiketi olarak kullanılacak
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            # CSV dosyasında "pandas.errors.ParserError: Error tokenizing data. C error: Expected 4 fields in line 8, saw 6" yaşanan hata için bir düzeltme
            try:
                # 21 satırı atlayarak başlıklar (wavelength ve absorbance) korunur son 10 satır nan ve infler çıkartılır
                data = pd.read_csv(file_path, skiprows=21, skipfooter=10, usecols=[0, 1], engine='python')
            except pd.errors.ParserError:
                print("Error parsing CSV. Please check the file format.")

            # Sınıf sütunun eklenmesi
            data['class'] = folder_name

            # Veriyi tek csv altında birleştirilmesi
            all_data = pd.concat([all_data, data], ignore_index=True)

# Sütun isimleri temize çekilir
all_data.columns = ["Wavelength", "Absorbance", "Class"]

""" # Veriyi gözden geçir
#print(all_data.head())

# Veri hakkında bilgi edin
#print(all_data.info())

# Eksik verileri kontrol et
# print(all_data.isnull().sum())  # Her sütundaki eksik değerlerin sayısını verir """

# Eksik verileri içeren satırları atma
all_data = all_data.dropna()  # NaN içeren satırları atar

""" # IQR yöntemi ile outlier'ları tespit etme
Q1 = all_data['Absorbance'].quantile(0.013)
Q3 = all_data['Absorbance'].quantile(0.987)
IQR = Q3 - Q1

# IQR'ın 1.5 katının dışındaki değerleri çıkarın
filtered_data = all_data[~((all_data['Absorbance'] < (Q1 - 1.4 * IQR)) | (all_data['Absorbance'] > (Q3 + 1.4 * IQR)))]

# Ölçeklendirilmiş veri yeni isimle kaydedilir
filtered_data['Absorbance_scaled'] = StandardScaler().fit_transform(filtered_data[['Absorbance']]) """

# Gruplama için "class" ve "Wavelength" sütunlarını kullanın
grouped_data = all_data.groupby(["Class", "Wavelength"]).mean()

# Absorbans ortalamalarını gösterin
#print(grouped_data["Absorbance"])



# Line plot ile ortalama absorbans değerlerini görselleştirme
sns.lineplot(data=grouped_data.reset_index(), x="Wavelength", y="Absorbance", hue="Class")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Average Absorbance (AU)")
plt.title("Average Absorbance by Wavelength and Class")
plt.show() 

# Bağımsız değişkenler olarak Wavelength ve Absorbance'ı kullanılır çünkü modelin Wavelength'e göre değişen Absorbance değerlerine göre eğitilmesi amaçlanır
X = all_data[['Wavelength', 'Absorbance']]
y = all_data['Class']

# Eğitim ve test setleri ayrılır seed olarak  42 kullanıldı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellik ölçekleme (yalnızca eğitim seti için)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# SMOTE ile veri setini dengeleme
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Sınıfları kodlama
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Özellikleri ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

""" # Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('gbc', GradientBoostingClassifier(random_state=42)),
        
    ],
    voting='soft'  # Soft voting kullanılır
)

# Voting Classifier'ı eğitin
voting_clf.fit(X_train_scaled, y_train_encoded)

# Modeli test edin ve doğruluğu ölçün
y_pred = voting_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Voting Classifier Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report) """

# Lojistik regresyon modeli ile iterasyon sayısını artır
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000) 

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(random_state=42)),
        ('lgb', lgb.LGBMClassifier(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=3)),
        ('gbc', GradientBoostingClassifier(random_state=42)),
    ],
    final_estimator=LogisticRegression()  # Üst model (final estimator)
)

# Stacking Classifier'ı eğitin
stacking_clf.fit(X_train_scaled, y_train_encoded)

# Modeli test edin ve doğruluğu ölçün
y_pred = stacking_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Stacking Classifier Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report)

#### Modelleri deneme

""" # SVM modelini eğitilir (RBF çekirdeği ile)
svm = SVC(kernel='rbf')  # veya 'linear', 'poly' vb.
svm.fit(X_train_scaled, y_train_encoded)

# Modeli test edilip doğruluğu hesaplanır
y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("SVM Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report)  """

""" # Karışıklık matrisi oluştur
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted SVM')
plt.ylabel('True')
plt.show() """

""" # Lojistik regresyon ile sınıflandırma
lr = LogisticRegression()
lr.fit(X_train_scaled, y_train_encoded)

# Modeli test edin ve doğruluğu hesaplayın
y_pred = lr.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Lojistik Regresyon Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report)

# Karışıklık matrisi oluştur
cm = confusion_matrix(y_test_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Logistic')
plt.ylabel('True')
plt.show() """

""" # Gradient Boosting modeli eğitimi
gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)  # 100 zayıf sınıflandırıcı
gbc.fit(X_train_scaled, y_train_encoded)

# Model doğrulaması
y_pred = gbc.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Gradient Boosting Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report) """

""" # Karışıklık matrisi oluştur
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Gradient Boosting')
plt.ylabel('True')
plt.show() """

""" # KNN modeli eğitimi
knn = KNeighborsClassifier(n_neighbors=3)  # K değeri (örneğin 3)
knn.fit(X_train_scaled, y_train_encoded)

# Model doğrulaması
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("KNN Doğruluk:", accuracy)

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report)  """


""" # Gaussian Naive Bayes modeli eğitimi
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train_encoded)

# Model doğrulaması
y_pred = gnb.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Gaussian Naive Bayes Doğruluk:", accuracy) 

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report) """

""" # XGBoost ile sınıflandırma
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_classifier.fit(X_train_scaled, y_train_encoded)  # Encoded veri ile fit

# Model doğrulaması
y_pred = xgb_classifier.predict(X_test_scaled)  # Modeli test et
accuracy = accuracy_score(y_test_encoded, y_pred)  # Doğruluk hesapla
print("XGBoost Doğruluk:", accuracy) 

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report) """

""" # LightGBM ile sınıflandırma
lgb_classifier = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_classifier.fit(X_train_scaled, y_train_encoded)

# Model doğrulaması
y_pred = lgb_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("LightGBM Doğruluk:", accuracy) 

# Model performansını değerlendirin
report = classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_)
print(report) """
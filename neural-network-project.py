import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Veri oluştur
X, y = make_classification(
    n_samples=1000, #Toplam örnek sayısı
    n_features=12,  # Toplam 12 özellik
    n_informative=12,   # Tamamı bilgi taşıyor
    n_redundant=0,  # Türetilmiş y
    n_repeated=0,   # Tekrar eden yok
    n_classes=3,    # 3 sınıf
    random_state=42 #Rastgelelik sabit hale geldi
)

# DataFrame'e çevir
feature_names = [f"Özellik_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["Sınıf"] = y


# Dengeli eğitim/test ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify parametresi sayesinde veri dengeli dağıtıldı
)

# One-hot encoding (Sınıf etiketlerini modelin anlayacağı hale getirdik)
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# Model oluşturma
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(12,)))  # 1. gizli katman
model.add(Dropout(0.25))                                    # dropout 1
model.add(Dense(32, activation='relu'))                    # 2. gizli katman
model.add(Dropout(0.25))                                    # dropout 2
model.add(Dense(3, activation='softmax'))                  # çıkış katmanı

# Modeli derleme
opt = Adam(learning_rate=0.005)  #Learning rate'i 0.005 olacak şekilde Adam optimizatörü eklendi
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',    # Kayıp fonksiyonu
    metrics=['accuracy']    # Doğruluk(Performans) ölçütü
)

# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',     # validation kaybı durumunda earlystopping devreye girecek
    patience=10,             # 10 epoch boyunca iyileşme olmazsa durdur
    restore_best_weights=True  # En iyi ağırlıkları geri yükle
)

#ModelChechkpoint
checkpoint = ModelCheckpoint(
    'en_iyi_model.keras',       # Modelin kaydedileceği dosya adı
    monitor='val_loss',      # validation kaybına göre en iyiyi seçecek
    save_best_only=True,     # Sadece en iyi (en düşük val_loss) olanı kaydet
    mode='min',              # val_loss küçükse daha iyi demek
    verbose=1                # Kaydederken bilgi ver

)

# Modeli eğitme
history = model.fit(
    X_train, y_train_cat,   #Eğitim verisi ve etiketleri
    epochs=30,  # 30 defa eğitim yapılacak
    batch_size=32, # Her eğitimde 32 örnek kullanılacak
    validation_split=0.2,   #%20'lik veri doğrulama için kullanılıyor
    verbose=1,  #ilerleme çubuğu gösterir
    callbacks=[early_stop, checkpoint]  #Earlystop ve checkpoint eklendi
)

#Metrikleri kaydetme
df1 = pd.DataFrame(history.history)
df1.to_csv('egitim_sonucu.csv', index=False)

#Modeli test etme
test_loss, test_acc = model.evaluate(   # model.evaluate: test verileriyle modeli değerlendirir
    X_test,       # Test veri girişleri (özellikler)
    y_test_cat,   # Test veri etiketleri
    verbose=0
)

# Test verisi üzerinde tahmin yap (etiket olarak)
y_pred_probs = model.predict(X_test) #sınıf olasılıkları
y_pred = np.argmax(y_pred_probs, axis=1) #gerçek sınıf etiketine çevir
y_true = np.argmax(y_test_cat, axis=1) #metrikleri hesapla yazdır

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Eğitim ve doğrulama loss grafikleri
plt.figure(figsize=(8, 5)) # Boyut ayarlama
plt.plot(history.history['loss'], label='Eğitim Kayıp (Loss)') #Eğitim verisi loss grafiği
plt.plot(history.history['val_loss'], label='Doğrulama Kayıp (Val Loss)') #Doğrulama verisi loss grafiği
plt.title('Eğitim vs Doğrulama Loss Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp (Loss)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Eğitim ve doğrulama accuracy grafiği
plt.figure(figsize=(8, 5))  # Boyut ayarlama
plt.plot(history.history['accuracy'], label='Eğitim Accuracy')          # Eğitim verisi accuracy
plt.plot(history.history['val_accuracy'], label='Doğrulama Accuracy')  # Doğrulama verisi accuracy
plt.title('Eğitim vs Doğrulama Accuracy Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (Accuracy)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Model tahminlerini alma
y_pred_probs = model.predict(X_test)  # Her sınıf için olasılık döner
y_pred_classes = np.argmax(y_pred_probs, axis=1)  # En yüksek olasılığa sahip sınıfı al

# Confusion matrixe gerçek sınıf etiketleri ekleme
y_true = np.argmax(y_test_cat, axis=1)  # One-hot aracılığyla gerçek sınıfları çıkar

cm = confusion_matrix(y_true, y_pred_classes) # Confusion matrix oluşturma

# Confusion matrix'i çizdirme
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])
disp.plot(cmap='Blues')  # Renk skalası: mavi tonları
plt.title("Test Verisi için Confusion Matrix")
plt.show()

print(f"Test doğruluğu: {test_acc:.4f}")

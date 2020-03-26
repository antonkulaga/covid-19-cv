
import os
import pandas as pd
import shutil
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import cv2
import os
pd.options.display.max_colwidth = 10000


def not_found(path: str) -> bool:
    return not os.path.exists(path)


base = "."
data_path = os.path.join(base, "data")
metadata = pd.read_csv(os.path.join(data_path, "metadata.csv"))

pa = metadata.where((metadata.view == "PA") & (metadata.modality == "X-ray"))[["finding", "path"]].dropna()
def get_image(image_path: str):
    image = cv2.imread(image_path)
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))

def image_loader(row):
    if(not_found(row["path"])):
        print("NOT FOUND")
    return get_image(row["path"])

pa["image"] = pa.apply(image_loader, axis=1)
pa["label"] = pa.apply(lambda row: "COVID-19" if row["finding"]=="COVID-19" else "other", axis=1)

normal_path = os.path.join(data_path ,"PA", "NORMAL")
normal_folder = os.listdir(normal_path)
normal_files = np.array([os.path.join(normal_path, d) for d in normal_folder])

nf = pd.DataFrame(data=normal_files, columns = ["path"] )
nf["finding"] = "NORMAL"
nf["label"] = "other"
normal_100 = nf.sample(100)
normal_100["image"] = normal_100.apply(image_loader, axis=1)


pa["image"] = pa.apply(image_loader, axis=1)
pa["label"] = pa.apply(lambda row: "COVID-19" if row["finding"]=="COVID-19" else "other", axis=1)


data_frame = pd.concat([normal_100, pa], keys=["image", "path", "label"])

lbs = data_frame["label"].to_numpy()

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = to_categorical(lb.fit_transform(lbs))

data =  np.array(data_frame["image"].to_list()) / 255.0 #scale intensities to the range [0, 255]

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2, stratify=labels, random_state=42)

#%%

INIT_LR = 1e-3 #learning rate
EPOCHS = 32
BS = 32 #batch size

#%%

# tougher than default
core_idg = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip = True,
    vertical_flip = False,
    height_shift_range= 0.05,
    width_shift_range=0.1,
    rotation_range=5,
    shear_range = 0.1,
    fill_mode = 'reflect',
    zoom_range=0.15
)



input_shape = Input(shape=(224, 224, 3))

#baseModel = VGG16(weights="imagenet", include_top=False,input_tensor= input_shape)
baseModel = DenseNet121( weights='imagenet',  include_top=False, input_tensor= input_shape)


#creating a head model on top of base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#freeze baseModel layers
for layer in baseModel.layers:
    layer.trainable = False


# compile our model
print("[INFO] compiling model...")
opt = tfa.optimizers.RectifiedAdam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])


# train the head of the network
print("[INFO] training head...")
H = model.fit_generator(
    trainAug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)


# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)



# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))


# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])


#%%


# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#%%

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

#%%

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")


#%%


# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

#%%

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVID-19 Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

#%%

# serialize the model to disk
print("[INFO] saving COVID-19 detector model...")
model.save("model", save_format="h5")
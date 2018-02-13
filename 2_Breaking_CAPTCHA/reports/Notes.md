# Breaking a CAPTCHA in 15 Minutes

So, this took me a lot longer to get to than I was planning on 🤦🏼‍♂️ Despite my best intentions and greatest ambitions, life continues to happen and usually take priority over projects I'm doing on the side. The irony is that this short little project is relatively simple, both [Adam Geitgey's](https://twitter.com/ageitgey) original project/code and my implementation of it.

For this project, and others like it, I've decided the best way to learn (and present what I've learned) is to go through the code, mostly line by line and explain what is going on. There are definitely opportunities to modify/improve the original code, use different training data, and test out different neural network implementations, but in the interest of time I'm just going to leave everything as is.

The tools being used for this project are Python 3, OpenCV, Keras, Tensorflow. Pretty much your standard deep learning stack with OpenCV being used for image augmentation and manipulation. OpenCV was used extensively in the first term of the Udacity Self Driving Car Nanodegree.

The CAPTCHA training data was "collected" by hacking around with the WordPress plug-in and outputting the CAPTCHA images along with their correct filenames. It's definitely possible to train a neural network on an image containing more than one letter/numeral, however, if it's possible to split them up the accuracy should increase and the training time decrease. This is exactly what was done using a few key functions in OpenCV.

The process for extracting the single letters, training the neural network, and using the model to solve CAPTCHAs is straightforward.

1. Extract single letters from the CAPTCHA images.

    `python extract_single_letters_from_captchas.py`

2. Train the neural network to recognize single letters.

    `python train_model.py`

3. Use the model to solve the CAPTCHAs.

    `python solve_captchas_with_model.py`

I'll go through each of these steps, explain the code and the problems that I encountered, and show the results.

## Extract single letters for the CAPTCHA images

`captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))`

```python
for (i, captcha_image_file) in enumerate(captcha_image_files):
    print("[INFO] processing image {}/{}".format(i + 1, len(captcha_image_files)))
    
    # Filename as text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]

    # Image augmentation
    image = cv2.imread(captcha_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find the contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
```

```python
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            letter_image_regions.append((x, y, w, h))
```

`letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])`

```python
    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        x, y, w, h = letter_bounding_box

        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)
```

## Train the neural network

```python
# loop over the input images
for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_to_fit(image, 20, 20)
    image = np.expand_dims(image, axis=2)

    label = image_file.split(os.path.sep)[-2]

    data.append(image)
    labels.append(label)
```

`data = np.array(data, dtype="float") / 255.0`

`(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)`

```python
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)
```

```python
model = Sequential()
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(32, activation="softmax"))
```

`model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])`

`model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)`

## Solve the CAPTCHAs
```python
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

model = load_model(MODEL_FILENAME)

captcha_image_files = list(paths.list_images(CAPTCHA_IMAGE_FOLDER))
captcha_image_files = np.random.choice(captcha_image_files, size=(10,), replace=False)
print(captcha_image_files)
```

```python
# Ask the neural network to make a prediction
prediction = model.predict(letter_image)

# Convert the one-hot-encoded prediction back to a normal letter
letter = lb.inverse_transform(prediction)[0]
predictions.append(letter)

# draw the prediction on the output image
cv2.rectangle(output, (x - 2, y - 2), (x + w + 4, y + h + 4), (0, 255, 0), 1)
cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
```

```python
cv2.imshow("Output", output)
cv2.waitKey(500) # Wait for a specified amount of time (in ms) before moving to the next image
```

```bash
/anaconda3/envs/ML3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using TensorFlow backend.
/anaconda3/envs/ML3/lib/python3.6/importlib/_bootstrap.py:219: RuntimeWarning: compiletime version 3.5 of module 'tensorflow.python.framework.fast_tensor_util' does not match runtime version 3.6
  return f(*args, **kwds)
Train on 29058 samples, validate on 9686 samples
Epoch 1/10
2018-02-08 15:14:08.402223: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
29058/29058 [==============================] - 41s 1ms/step - loss: 0.2413 - acc: 0.9413 - val_loss: 0.0226 - val_acc: 0.9950
Epoch 2/10
29058/29058 [==============================] - 44s 2ms/step - loss: 0.0160 - acc: 0.9963 - val_loss: 0.0140 - val_acc: 0.9968
Epoch 3/10
29058/29058 [==============================] - 46s 2ms/step - loss: 0.0062 - acc: 0.9983 - val_loss: 0.0081 - val_acc: 0.9977
Epoch 4/10
29058/29058 [==============================] - 40s 1ms/step - loss: 0.0052 - acc: 0.9986 - val_loss: 0.0054 - val_acc: 0.9988
Epoch 5/10
29058/29058 [==============================] - 38s 1ms/step - loss: 0.0022 - acc: 0.9993 - val_loss: 0.0115 - val_acc: 0.9975
Epoch 6/10
29058/29058 [==============================] - 37s 1ms/step - loss: 0.0069 - acc: 0.9985 - val_loss: 0.0076 - val_acc: 0.9979
Epoch 7/10
29058/29058 [==============================] - 39s 1ms/step - loss: 0.0026 - acc: 0.9994 - val_loss: 0.0129 - val_acc: 0.9971
Epoch 8/10
29058/29058 [==============================] - 38s 1ms/step - loss: 0.0050 - acc: 0.9984 - val_loss: 0.0177 - val_acc: 0.9947
Epoch 9/10
29058/29058 [==============================] - 41s 1ms/step - loss: 0.0011 - acc: 0.9997 - val_loss: 0.0047 - val_acc: 0.9991
Epoch 10/10
29058/29058 [==============================] - 42s 1ms/step - loss: 2.3045e-05 - acc: 1.0000 - val_loss: 0.0038 - val_acc: 0.9991
The elapsed training time is:  429.6541633605957
```
*note: make sure you have h5py installed before you train the model, otherwise you're going to be doing all of that training again.*
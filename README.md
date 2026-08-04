Model is ready, need to fit in CI/cd

Phase 1 — Build a reproducible training framework
1) Dataset download (automatic)
2) Data ingestion
3) Data validation
4) Data transformation
5) Model training
6) Model evaluation
7) Save best model and metrics

Phase 2 — Build the inference framework
1) Model loading
3) Image preprocessing
4) Single-image inference
5) Batch inference

Phase 3 — Real-time system
1) Face detection
2) Webcam stream
3) Emotion prediction
4) Visualization
5) FPS and confidence display

Phase 4 — Production engineering
1) Docker
2) GitHub Actions (CI)
3) Unit tests
4) Configuration management
5) Logging
6) Experiment tracking (optional: MLflow/W&B)

Phase 5 — Deployment
1) Streamlit UI
2) FastAPI backend or Flask
3) Cloud deployment (Render)



┌────────────────────────────────────────────┐
│ Real-Time Emotion Detection                │
├────────────────────────────────────────────┤
│                                            │
│   📷 Webcam                                │
│                                            │
│   ┌──────────────────────────────┐         │
│   │ 😊 Happy                     │         │
│   │ Confidence: 96.7%            │         │
│   │ FPS: 29.8                    │         │
│   │ Latency: 8.3 ms              │         │
│   │                              │         │
│   │          Face                │         │
│   └──────────────────────────────┘         │
│                                            │
│ Last 10 Predictions                        │
│ Happy Happy Happy Happy Happy Happy ...    │
│                                            │
│ Session Statistics                         │
│ Happy     ██████████ 67%                   │
│ Neutral   ████       20%                   │
│ Sad       ██         13%                   │
└────────────────────────────────────────────┘


04-08-2026

The accuracy issue, at the stage which the project is there is been an issue which is the model is not predicting the correct features or emotions on the test data, when being prompted to do the batch inference in the inference pipeline

the test images data has 2 images which is sad_1 and happy_2 whose names specify their emotions respectively.

what is happening is that for the image happy_2 the model is predicting the emotion as happy with confidence of ~44.4% which is fine and can be improved

but for the image sad_1 the model predicts the emotion as happy with over ~90% confidence which is concerning,

I have tried to feed the model the cropped images because the dataset RAF-DB has a lot of cropped dataset but still the model shows the same.

fixes/experiments:

There are a lot of angles which can throw a ray of hope but the combinations to achieve this scare me more:

1. we need to alter the database where we add some more images of selfies because the images in the existing dataset and the selfies are a bit different w.r.t lighting, angle etc. so we might have to include some images while training and check if the model is picking it up after that.

2. for the model weights we are using cross-entropy as of now instead of that we need to compute the class weights inversely proportional to class frequencies so the model penalization is higher when it misclassifies a minority class, or maybe we can shift to focal loss and check them

3. in the inference pipeline instead of bounding box cropping, we might have to use facial landmarks to align the eye levels, and also what I am thinking is that as we are using bounding boc=x cropping the model is picking up the chin lines and other features and misclassifying it, we have to find a workaround for this.

4. Unfreeze Deeper Layers / Fine-tune Backbone:If using a pretrained backbone (e.g., ResNet18/50), ensure the final 1 or 2 convolutional blocks are unfrozen during training with a lower learning rate ($10^{-4}$ or $10^{-5}$) so high-level facial feature representations adapt to real-world domain nuances.

5. Post-Processing Confidence Thresholding:Since the model has a high prior bias toward happy, set a higher confidence threshold (e.g., >60–70%) before accepting happy as the prediction; otherwise, fall back to evaluating secondary class probabilities (like neutral or sad).
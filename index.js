const classifier = knnClassifier.create();
const webcamElement = document.getElementById("webcam");
let net;

async function app() {
    console.log("Caricando mobilenet...");

    //Loading model
    net = await mobilenet.load();
    console.log("Modello caricato.");

    // Prediction making/Fetchting video feed from the web camera
    const webcam = await tf.data.webcam(webcamElement);
    // Reading the image from the webcam and associating it with a class index
    const addExample = async classId => {
        // Capturing the image
        const img = await webcam.capture();

        // Activating Mobilenet and passing it to the KNN classifier
        const activation = net.infer(img, true);

        // Passing the activation
        classifier.addExample(activation, classId);

        // Releasing memory
        img.dispose();
    };

    // When clicking a button, add an example for that class
    document.getElementById("class-a").addEventListener( "click", () => addExample(0));
    document.getElementById("class-b").addEventListener( "click", () => addExample(1));
    document.getElementById("class-c").addEventListener( "click", () => addExample(2));
    document.getElementById("class-d").addEventListener( "click", () => addExample(3));
    
    while (true) {
        if (classifier.getNumClasses() > 0) {
            const img = await webcam.capture();

            // Getting the activation from Mobilenet from the webcam
            const activation = net.infer(img, "conv_preds");

            // Returning the most likely class and confidence from the classifier module
            const result = await classifier.predictClass(activation);

            const classes = ["Rosso", "Verde", "Arancione", "Blu"];
            document.getElementById('console').innerText = `
              prediction: ${classes[result.label]}\n
              probability: ${result.confidences[result.label] * 100}%
            `;

            // Releasing memory
            img.dispose();
        }

        await tf.nextFrame();
    }

}
app();
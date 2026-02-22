# FGSM Attack: When AI Starts Hallucinating

Hi there! 👋 As someone deeply passionate about the intersection of Artificial Intelligence and Cybersecurity, I've always been fascinated by the vulnerability of deep neural networks. This repo is the result of one of my explorations: how to fool a state-of-the-art AI by subtly modifying a few pixels using math? 🤔

Here, I implement the **FGSM (Fast Gradient Sign Method)** attack on a ResNet50 model using PyTorch. It's crazy to see that by injecting noise specifically calculated from the gradients (often imperceptible to the human eye), you can force a highly performant model to make a completely wrong prediction! 🤯🛡️

🎥 **Watch it in action!** A demo video has already been published on my LinkedIn page. You can check it out directly by following this link: [Watch the Demo on LinkedIn](https://www.linkedin.com/posts/abdoul-gafarou-zoungrana-486280323_intelligenceartificielle-machinelearning-activity-7428907186644492288-UYjE?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFG7UqYBp5RJm-fLA5tLnFigcgVh0cdFuRo)

## 💡 Why am I so passionate about this?
We often tend to see AI as an infallible black box. But when you put on your "cyber" hat, you realize that these systems have gaping flaws. Using the loss function and backpropagating the gradients not to train the model, but to create **adversarial examples**, is a bit like reverse-engineering the machine's "perception". It's exactly this kind of challenge that pushes me to dig deeper and deeper into AI security! 

## What the code does
This script (designed to be run quickly on Google Colab) allows you to:
1. Load a pre-trained **ResNet50** model on ImageNet.
2. Upload an image of your choice.
3. Calculate the gradient of the loss with respect to the input image itself.
4. Create the perturbation (the famous noise) and add it to the original image with an epsilon factor ($\epsilon = 0.05$).
5. Display a nice graph comparing the original image, the generated noise, and the attacked image with their respective predictions. 🎯💥

If you use it on Google Colab (recommended given the `google.colab` import), the libraries are generally already installed. Otherwise, you will need:
* `torch` & `torchvision`
* `Pillow`
* `matplotlib`
* `requests`

---
*Feel free to fork, play around with the epsilon value, or test it on other models to see how robust they are (or aren't)!* 😉

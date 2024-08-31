# Fblthp Foundries

Magic: The Gathering project for generating custom cards. We support a complete generative pipeline including card text generation, art synthesis, and Photoshop image rendering.

Much of the pipeline involves 'hot-swappable' components based on the preferences of the user. For example:

## Text generation:

We support two pathways for text generation. 

**gpt-4o-mini**: We use multiple requests to the OpenAI API to produce card themes and names, fine-tune the text output, and then we parse it and feed it on to the next step. *Estimated cost per card < $0.001*

**Fine-tuned GPT-2 model**: This model was trained on a corpus of existing MTG cards, which were preprocessed into an XML-like structure to encourage the model to better learn the card structure. We also created a custom tokenization process. The model is used to generate card text which is then parsed and fed on to the next step. *$0/card, assuming locally ran*

## Image prompt generation:

We support two pathways for image prompt generation:

**gpt-4o-mini**: We use the card text as input to the chatbot and ask it to give us an art prompt that can be used to generate card art. *Estimated cost per card < $0.001*

**local**: We use a script to extract relevant parts of the card and combine them into a prompt that can be used to generate card art. *$0/card*

## Image generation:

We support two pathways for image generation.

**DALL-E**: An image is requested from DALL-E via the OpenAI API with the previously generated prompt. *Cost per card: $0.04 for 1024x1024, $0.08 for 1792x1024*

**Stable Diffusion 3**: An image is generated locally using the Stable Diffusion 3 open-source weights. *$0/card* (local GPU is ideal)

## Card rendering:

**Proxyshop**: *Proxyshop* photoshop plugins and templates are used to render the card's final image.


Technologies: PyTorch, fine-tuned GPT-2 model, Stable Diffusion 3, OpenAI API, Proxyshop (a photoshop extension for generating card images)
Serving live image delivery over the internet (not)


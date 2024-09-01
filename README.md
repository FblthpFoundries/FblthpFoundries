# Fblthp Foundries

**Fblthp Foundries** is an innovative Magic: The Gathering project dedicated to generating new and unique cards using a combination of AI models. Our pipeline covers every step of the process, from text generation to final card rendering, offering customizable options based on user preferences.

## Generative Pipeline Overview

### Text Generation

We offer two pathways for generating card text:

- **GPT-4o-mini:** This method leverages OpenAI's API to generate card themes, names, and fine-tuned text outputs. The structured text is parsed and prepared for the next stage in the pipeline.
  - *Estimated cost per card: < $0.001*

- **Fine-tuned GPT-2 Model:** Trained on a comprehensive corpus of existing MTG cards, this model employs custom tokenization and XML-like preprocessing to accurately generate card text. The output is then parsed and forwarded to the next stage.
  - *Cost per card: $0* (Assuming local execution)

### Image Prompt Generation

Image prompts are generated using two pathways:

- **GPT-4o-mini:** Using the generated card text, we ask GPT-4o-mini to create a detailed art prompt suitable for generating card art.
  - *Estimated cost per card: < $0.001*

- **Local Script:** This method extracts relevant details from the card and assembles them into a prompt for art generation.
  - *Cost per card: $0* 

### Image Generation

We support two methods for generating card art:

- **DALL-E:** High-quality images are generated via the OpenAI API using the previously created art prompt.
  - *Cost per card: $0.04 for 1024x1024, $0.08 for 1792x1024*

- **Stable Diffusion 3:** Using open-source weights, this method generates images locally, ideal for those with access to a capable GPU.
  - *Cost per card: $0* (Local GPU required)

### Card Rendering

The final step in the pipeline involves rendering the card:

- **Proxyshop:** We use Photoshop plugins and templates from Proxyshop to create the final card image. This method requires a valid Photoshop installation but is otherwise free.
  - *Cost: Free* (with Photoshop)

## Additional Features

We're also developing a Discord bot and web server to further streamline the card generation process and provide additional functionalities.

## Future Goals

Our ongoing efforts aim to enhance the balance of the generated cards and boost the overall creativity of the AI models.

## Technologies

- PyTorch
- Fine-tuned GPT-2 model
- Stable Diffusion 3
- OpenAI API
- Proxyshop (Photoshop extension for card rendering)

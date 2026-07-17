# episodic-latent-reasoning - archive

This folder contains code and attempted designs that was originally intended to be a part of the main project structure, but ultimately decided to be scrapped for the time being. For implementations intended to be surfaced from this project, head to the `src` folder. Notably some of these might be useful for future development. Explanations on what each folder is and why it is left behind are provided below

## TextJEPA

Not to be confused by the TextJEPA implementation of [Bui et al. - Speaking in Words, Thinking in Logic: A Dual-Process Framework in QA Systems](https://arxiv.org/abs/2507.20491v1). This folder was originally intended to build the pretrained encoder used for R-JEPA of this project using similar approach implemented in I-JEPA [Assran et al. - Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), but for text utilizing word-bounded span masking

### Limitations and Challenges

- **Words are dynamic length in tokens**: Dropping a word in text is not equivalent to dropping a patch in images as a single word consists of multiple tokens, while a patch is represented in a single token
- **Efficiency in Implementation**: Using words as the block level for masking would cause jagged tensors as $x$ words in $Block_a$ might be different in length to another in $Block_b$, This technically can be solved using padding but should look for more efficient approach to model the masking for the task instead.
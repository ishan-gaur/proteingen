Idea is to make this setup as functional as possible so that it
nicely mirrors the mathematical formulation.

Main types:
- integrator: euler and exact
    - takes sequence, get_transition_probs, (euler takes transition_rate as input to get the sampler that takes the sequence and get_transition_probs)
- sequence is a list of strings of sequences
    - It is assumed that the predictive model uses the same tokenizer as the generative model--so masking etc. will be handled internally
- get_transition_probs takes a sequence as input and returns a tensor of logits for each position
    - the shape of the output logits must match the input--how do you enforce that?
        - e.g. for a conditional model you need a factory function that takes the conditioning information as input and returns a function that will load up the conditioning information appropriately and just provide a seq->seq interface for the rest of the code
        - I feel like an important downside of this for lay users is that the programming will be counterintuitive. It'll even be intimidating for regular but not try-hard programmers.
        - So like for guidance we have a bayes_rule generative_model, "get_conditional_forward" in esm_cath/model.py:ESMCath will also return something of type generative_model
    - these will be of type generative_model
    - an unconditional ESM's forward function logits (appropriately wrapped) is a valid generative model
    - For example, the bayes_rule generative_model will be constructed by taking a generative model (of type generative_model)
      and a predictive model as input
    - TAG will be another generative_model
    - In this way, e.g. multiple classifiers can be layers by wrapping them in multiple generative models
- transition_rate will have two functions--a time-conditional rate and an unconditional sampler
    - the time-conditional rate will be used to compute rates for TAG
    - the unconditional sampler will be used to sample noise schedules for training
- Predictive model takes one-hot encoded sequences as input so that TAG works as expected
    - So we need to adjust ESM accordingly if it is to be a backbone for linear probes
        (linear probs will take an "embed" function as input, which will be the ESM embedding for ESM, etc.)

DUMP OF BIG PICTURE TODOS:

Where do we store models? Only use ones on HF??
- PMPNN, ESMC, ESM3, 

Think we should implement the paper experiments--or at least the tricks needed to make them work
We should also make it easy to do diagnostics like check the likelihoods
We should also include code to use FT so it's easy for people to do
And we can make our recommendation to do both or at least do partial unmasked generation

I should've maybe made it so that you could use HF models no???
I guess you can--just have to wrap them with a generative model--which is just adding a logitformatter
That would be the real win
How do packages get there--maybe ask Tristan



DATASET:
# Guidance dataset build on top of regular model training dataset, which is linked to its sampler
# Conditional training and guidance training can use the same dataset, they just treat different parts as inputs and output

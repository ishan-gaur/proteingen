TODO[pi] integrate with https://github.com/COLA-Laboratory/GraphFLA to get the different fitness landscapes
- Also FLIP and SaProtHub
- Models: Progen, Dayhoff, Evodiff, PMPNN, METL, SaProt

- [ ] Test guided sampling
    - [ ] Get a stability guided generation demo
        - [ ] ESM3IF
        - [ ] Integrate the stability mpnn
        - [ ] Setup his code into a minimal demo
    - [ ] Implement Tanja work as pilot for classifier stuff
        - [ ] How do you support both cdf below and above with just values
        - [ ] Make a NormalPDF model (stub for RegressionEnsemble, which inherits from this)
        - [ ] Add the ClassificationPredictiveModel
        - [ ] Define an embedding model protocol so that you can just pass the model class to the linear probe and it can use it
- [ ] Classifier training script
    - [ ] Dataset interface with collators
    - [ ] Add a classifier to the predictive models
- [ ] Add euler integrator and test tag
- [ ] These context managers feel a bit stupid, seems fine to just condition in place and have to call unconditional when you want that
    - [ ] If we keep it, just add a wrapper for the variable that auto adds the method to the class
- [ ] Move models to HF
- [ ] Add a TAGModel for getting likelihoods
- [ ] Do an autoregressive example and do twisted SMC

# TO TEST
- Guided Sampling
    - That the TokenizerTranslator actually does map between the PredictiveModel and GenerativeModel

def model(data):
    num_topics = 2
    nrow = data.shape[1]
    ncol = data.shape[2]

    with pyro.plate('topic', num_topics):
        # sample a weight and value for each topic
        topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / num_topics, 1.))
        topic_concentrations = pyro.sample("topic_values", dist.Gamma(2 * torch.ones(nrow,ncol), 1/3 * torch.ones(nrow,ncol)).to_event(2))
        print('topic weights', topic_weights.shape)
        print('topic values', topic_concentrations.shape)

    with pyro.plate('participants', data.shape[0]):
        # sample each participant's idiosyncratic topic mixture
        participant_topics = pyro.sample("participant_topics", dist.Dirichlet(topic_weights))
        print('participant topics', participant_topics.shape)
        
        transition_topics = pyro.sample("transition_topics", dist.Categorical(participant_topics),
                                        infer={"enumerate": "parallel"})
        print('topics')
        print('transition topics', transition_topics.shape)

        out = dist.Dirichlet(topic_concentrations[transition_topics]).to_event(1)
        print('observation batch:', out.batch_shape)
        print('observation event:', out.event_shape)
        print('data', data.shape)
        data = pyro.sample("obs", dist.Dirichlet(topic_concentrations[transition_topics]),
                    obs=data)
        print(data)

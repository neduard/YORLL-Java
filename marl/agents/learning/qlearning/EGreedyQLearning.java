package marl.agents.learning.qlearning;

import marl.agents.selection.Argmax;
import marl.agents.selection.EGreedy;
import marl.environments.State;
import marl.utility.Config;

public class EGreedyQLearning<S extends State<S>>
    extends DiscreteQLearning<S>
{
    /**
     * Uses Epsilon greedy exploration
     */
	private EGreedy egreedy_;
	
	public EGreedyQLearning(Config cfg)
	{
		super(cfg);
		egreedy_ = new EGreedy(cfg);
	}

	@Override
	public int select(S state)
	{
		double[] qValues = qTable_.get(state);
		if( evaluationMode_ )
            return Argmax.select(qValues);
		else
		    return egreedy_.select(qValues);
	}
	
    /**
     * Decreases the value of epsilon in the EGreedy selection algorithm.
     * @param int episodeNo The episode number
     */
	public void decreaseEpsilon(int episodeNo)
	{
		egreedy_.decreaseEpsilon(episodeNo);
	}

}

package marl.agents.learning.sarsa;

import marl.agents.selection.Argmax;
import marl.agents.selection.EGreedy;
import marl.environments.State;
import marl.utility.Config;

public class EGreedySarsa<S extends State<S>>
    extends DiscreteSarsa<S>
{
	private EGreedy egreedy_;
	
	public EGreedySarsa(Config cfg)
	{
		super(cfg);
		egreedy_ = new EGreedy(cfg);
	}

	@Override
	public int _select(S state)
	{
		double[] qValues = qTable_.get(state);
		if( evaluationMode_ )
            return Argmax.select(qValues);
		else
		    return egreedy_.select(qValues);
	}
	

    /**
     * Decreases the value of epsilon in the EGreedy selection
     * algorithm.
     * @param episodeNo The episode number
     */
	public void decreaseEpsilon(int episodeNo)
	{
		egreedy_.decreaseEpsilon(episodeNo);
	}

}

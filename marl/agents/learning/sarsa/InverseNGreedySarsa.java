package marl.agents.learning.sarsa;

import marl.agents.selection.Argmax;
import marl.agents.selection.InverseNGreedy;
import marl.environments.State;
import marl.utility.Config;

public class InverseNGreedySarsa<S extends State<S>>
    extends DiscreteSarsa<S>
{
	private InverseNGreedy egreedy_;
	
	public InverseNGreedySarsa(Config cfg)
	{
		super(cfg);
		egreedy_ = new InverseNGreedy(cfg);
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

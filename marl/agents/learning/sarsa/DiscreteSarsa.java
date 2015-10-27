package marl.agents.learning.sarsa;

import marl.agents.learning.LearningAlgorithm;
import marl.agents.learning.qlearning.DiscreteQTable;
import marl.environments.State;
import marl.utility.Config;

public abstract class DiscreteSarsa<S extends State<S>>
	extends LearningAlgorithm<S>
{
    protected DiscreteQTable qTable_;    // The Q table
    protected double         alpha_;     // The learning rate
    protected double         gamma_;     // The discount factor
    protected int            nActions_;  // The number of actions
    
    protected S              potentialState_;
    protected int            potentialAction_;
                                         // The next action
	
	
	public DiscreteSarsa(Config cfg)
	{
		alpha_ = cfg.getDouble("alpha");
		gamma_ = cfg.getDouble("gamma");
        
        {
        	try{ 
        		int nStates = cfg.getInt("num_states");
        		qTable_ = new DiscreteQTable(nStates);
        	} catch(NumberFormatException ex) {
        		qTable_ = new DiscreteQTable();
        	}
                
            qTable_.reset();
        }
        
        potentialState_  = null;
        potentialAction_ = -1;
	}
	
	
	@Override
	public final int select(S state) {
	    // make sure that the select function aligns with the update function
	    if( potentialState_ == null || !state.equals(potentialState_) )
	        return _select(state);
	    else
	        return potentialAction_;
	}
	abstract protected int _select(S state);
	


	/**
	 * Note: Select *must* have been called before update otherwise action_
	 *       will not have been set.
	 */
	@Override
	public void update(
			S curState, S newState,
			int action, double reward)
	{
	    if( !evaluationMode_ ) {
    		double[] curQValues = qTable_.get(curState);
    
            // Get the old and max Q values
            double   oldQ, newQ, nextQ = 0.0;
    		
    		if( newState != null ) {
    		    double[] newQValues = qTable_.get(newState);
    		    potentialState_     = newState;
    		    potentialAction_    = _select(newState);
    	        nextQ               = newQValues[potentialAction_];
    		}
    		else
    		    potentialState_     = null;
            
            oldQ = curQValues[action];
            
            newQ = oldQ + (alpha_ * (reward + (gamma_*nextQ) - oldQ));
            
            qTable_.put(curState, action, newQ);
	    }
	}
	
	

    /**
     * Inform the SARSA of the available actions for the
     * specified state. This should be called as and when needed
     * to inform the learning algorithm that it is looking up a
     * state the the specified number of actions.
     * 
     * @param nActions The number of actions current available
     */
    @Override
    public void inform(int nActions)
    {
        super.inform(nActions);
        qTable_.inform(nActions);
    }

}

# Mistakes throughout implementing the code

### 1. Wrong usuage of torch function
While debugging line by line and spending so much time, I found out that unexpected broadcasting was happening in my code.
I wasn't familiar with a function `torch.gather(input, dim, index)` and input the wrong shaped tensor. 
If input was shape of (N, A) (where N is the batch size and A is the number of possible actions), 
dim should be 1 as we want to gather elements on action axis, 
and index should be a shape of (N, 1).
However, I set the shape of index as (1, N) which resulted in unexpected broadcasting rule.
Let's make sure we are using torch function in a correct way!


### 2. Dying ReLU
Running for near 3 million steps, the agent did not show a sign of learning.
When I checked target values of a batch, their values were all the same.
This means no matter what the input is, agent cannot tell the difference. 
Thus, the agent performed only single action throughout an episode when the action is chosen greedily.

After more debugging, I realized only the bias parameters of the model was learning.
To confirm that, I set `bias=False` for all the layers, and was able to check gradient suddenly drops to zero.
This means the model is not learning at all.

Searching for the solution, I found out this phenomenon is called __Dying ReLU__.
Any input to the weight returns negative values, which results in zero gradient.
One remedy to this is to reduce learning rate.

[https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)


### 3. Train takes long
I was able to get an insight on what we usually overlook by the success of some RL by reading the article called, [Deep Reinforcment Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html).

One thing about Reinforcement Learning is that it is sample inefficient.
Refer to the accumulated reward graph shown in the linked article.
It takes __millions of frames__ to perform decent, and until then the accumulated reward is very low.
One million frames means 4 hours when agent learns on MacBook Pro GPU.

Many of the tutorial codes does not learn from pixels, thus learn so much faster.
Do not let them blind you from the difficulty of training the agent from unformatted data.

__So be patient.__


### 4. Misunderstanding frame skip
The concept of frame skipping is quite confusing because the original DQN paper states that they skip frames in which agent repeats the same actions, and that they take last four frames to feed into the function approximation model.

As I understood it as taking the last four frames of the same action.
Then agent was stuck in the suboptimal behavior.
![bad_frameskip](img/(bad_frameskip)ALE%3APong-v5.png)

Reading through some related discussion in [reddit](https://www.reddit.com/r/reinforcementlearning/comments/fucovf/confused_about_frame_skipping_in_dqn/), I found an amazing [blog](https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/) that breaks down the confusion of frame-skipping

So, the frame skipping and feeding last four frames are tagent. 
Thus, each frame that's fed into the function approximator should be the observation of different actions.
Frame skipping being fixed, the agent finds actions to beat the game.
![good_frameskip](img/ALE%3APong-v5%20copy.png)
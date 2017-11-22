---
permalink: /rnn-lab-autonomous/
---

## Recurrent Neural Networks (Autonomous Laboratory)

This session is for exploring the effects of the different elements of 
RNNs using a specific dataset. In this case we are going to extend the Wind
prediction example that we used in the previous session.

Deliver your report before **December 22th** using the raco.


In the original example we predicted one step of the future wind (next 15 minutes)
using a window of previous measurements. We can extend the problem in different
ways:

### Complementary variables

In the original experiment we used only the **wind speed** as input data. The
measurements include also the **air density**, the **temperature** and the
**presure**.

Do the following:

* Modify the arquitecture adequately for the input
* Study how the MSE changes if you change the size of the units (more/less memory),
the dropout and the length of the input window.

### Complementary sites

In the original experiment we predicted the **wind speed** using the data from
one site, the dataset includes three additional sites that are geographically
close (they are in the vertices of a 2 km square). We can leverage the information
of the *wind speed* from the other sites.

Do the following:

 * Use the previous architecture (you have now also 4 variables, 4 wind speeds) and study how
 the MSE changes if you change the size of the units (more/less memory),
the dropout and the lenth of the input window.
 

### Multi step prediction

We can obtain a multi step prediction using the original model simply by adding the
value predicted to the current set of measurements and discarding the oldest one, 
just like in the text generation example.

The new prediction will be two steps ahead, $n-1$ measurements will be actual
observations and one will be a predicted one.

Perform some experiment using this method and observe:

* How the MSE degrades the more steps in the future we predict
* How the MSE changes if we extend the input window

For comparing the MSE you should separate the predictions according on how far
are the measurements. That is, compute the MSE for the one step prediction, then
for the second step prediction, and so on. 

### Sequence to sequence prediction

An alternative to shifting the input and adding the predictions to have a multi
 step prediction would be a network that links a window of measurements to a
window of predictions.

Do the following:

* Adapt the sequence to sequence arquitecture from the summation example to this
task
* Study how the MSE of the predictions changes with the length of the predicted
sequence


##  What you have to do

 Make a plan of the experiments that you want to perform. This does not mean that
 the plan is written on stone, revise the plan as you see fit given the results or
  extend/change the experiments when you judge that it is interesting given the results.

  Execute the experiments and record all the results and decisions that you make

 Write a report explaning:

* The plan of the experiments that you have performed
* What were you expecting with each experiment and what you actually obtained 
(eg, you though that some conditions of the experiment would improve the results 
because of *insert reason here* and the results were the same/worse)
* The interpretation of the results of the experiments:

	* what did work, what did not work (eg: to increase the number neurons up to a number does not give any gains, to use dropout makes think worse consistently, to predict more than one step in the future works with these data up to a limit)

	* what set up/architecture/data improves the baseline (persistence), some intuition about why?

Use statistics and graphics to support your explanations.


Deliver your report before **December 22th** using the raco.



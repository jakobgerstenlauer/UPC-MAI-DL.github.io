---
permalink: /rnn-lab-autonomous/
---

This session is for exploring the effects of the different elements of the
RNN using a specific dataset. In this case we are going to extend the Wind
prediction example that we used in the previous sessions.

In the original example we predicted on step of the future wind (next 15 minutes)
using a window of previous measurements. we can extend the problem in different
ways:

### Complementary variables

In the original experiment we used only the **wind speed** as input data. The
measurements include also the **air density**, the **temperature** and the
**presure**.

Do the following:

* Modify the arquitecture adequately for the input
* Study how the MSE changes if you change the size of the units (more/less memory),
the dropout and the lenth of the input window.

### Complementary sites

In the original experiment we predicted the **wind speed** using the data from
one site, the dataset included three additional sites that are geographically
close (they are in the vertices of a 2 km square).

Do the following:

 * Use the previous architecture (you have now also 4 varibles) and study how
 the MSE changes if you change the size of the units (more/less memory),
the dropout and the lenth of the input window.
 * **Optional:** Experiment what happens if you  include as initial
 Pooling layer (average, maxpool) so instead of the wind in the site you have
 the maximum/average wind speed of the four sites as input.

### Multi step prediction

We can obtain a multi step prediction using the same model simply adding the
value predicted to the current set of measurements and discarding the oldest one.

The new prediction will be two steps ahead, n-1 measurements will be actual
observatrions and one will be a predicted one.

Perform some experiment using this method and observe:

* How the MSE degrades the more steps in the future we predict
* How the MSE changes if we extend the input window

For comparing the MSE you should separate the predictions according on how far
are the measurements. That is, compute the MSE for the one step prediction, then
for the second step prediction, and so on. The number of elements in the MSE
computation must be the same to be able to compare.

For example, if you compute the one and two step predictions for a set
consecutive windows, the first window will predict instants $t$ as first step
prediction and $t+1$ as second step prediction, but you will have no second step
prediction for instant $t$.

### Sequence to sequence prediction

An alternative to shifting the input and adding the predictions to have a multi
 step prediction would be a network that links a window of measurements to a
window of predictions.

Do the following:

* Adapt the sequence to sequence arquitecture from the summation example to this
task
* Study how the MSE of the predictions changes

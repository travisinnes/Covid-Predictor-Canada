Purpose: To create a simulator that would predict the spread of COVID-19 based on current trends as well as different possible measures of government disease control ie: social distancing.

Project: 
With our code we use equations to model our data and fit it to real data of COVID-19 to then predict the course of the disease in the different provinces and territories over the coming months.  At the very beginning of our code we have all our necessary imports.  We then create a matrix stored in the variable ‘L’ that models how the various provinces and territories are connected.  A value of 1 is assigned to provinces that either border each other or have direct flights between them, or 0 if it has neither of these.  

To model the spread of the disease, we consider 4 groups of population:
1. S - Susceptible, healthy people that have no virus 
2. E - Exposed, people who contracted the virus but are not sick
3. I - Infected, people who were exposed and became sick
4. R - Recovered, those who recovered (or died)
 
The typical flow goes from 1 to 4.  We use 7 parameters that control the rate of the spread of the disease.  We create a function where we assign our various parameters (α, β, γ, μ, ks, ke, and ki) a spot in an array called ‘theta’.  We then assign variables for the derivatives with respect to time of the susceptible, exposed, and infected populations and create an array of 13 zeros, where each place represents a province or territory.  We then create the equations for each derivative, which takes the parameter values as well as values of different arrays for the different population groups that I will get to.  This function returns three different arrays for the derivatives of the different population groups, which each spot in the array representing the derivative of a given province or territory.
 
We then create another function where we create three different arrays of zeros for the susceptible, exposed, and infected populations, with each spot representing that of a given province.  We then create a matrix for each of these populations, to store the population values over time, where once again, each column represents a province and each row down is the next step forward in time.  The first row will have the initial population values.  We then create a for loop and if/else if statement that calls upon our previous function to give us the derivative values of each population at each step.  Note that we use different parameters after a certain date once protective measures have been put in place, which is what the if/else if statement accounts for.  We then obtain the values for each population by adding the derivative multiplied by the change in time to the previous population value, and store this value in the matrix for each population group, working our way across the provinces and down the rows with time.  This function returns these three matrices.
 
For our solution to finding parameters past this point we tried two different methods:
  Guess and check: 
    The parameters are estimated based on assumptions of the activity of two different schemes. One reflects the outbreak of COVID with     protective measures taken and the other without. This method is used without the use of the loss coefficient and gradient descent.  
    This method also proves to be less accurate as it scales as parameters may not perfectly reflect the situation.
 
  Loss & parameter optimization:
    Initially we define the SEI models, but we also define a loss function. This function is used to compare the results of our model to      the real data.  It takes in two arrays, one for the real infected population, and one for our model’s infected population and            returns  a value that tells us how accurate we are.  From here we have another function that scans through the various combinations      of parameter values to find one that gives us the lowest loss value. This in turn would give us our optimal parameter values.            Although this solution will yield us the correct results, the problem comes from the number of loops we must run through to find        the parameters. Upon testing, we let the program run for a half-hour without success.
 
Finally, our last block of code is that which produces the graphs for each province.  It runs through the three arrays, working down the rows staying in a given column (one province) and plots each of these three lines.  We then printed the max people infected and percent infected, as well as the number of people infected on April 21st and percent by using the max function when calling the matrix and calling the given data point in the matrix that represents April 21st.

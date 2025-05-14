# cct-midterm
Cultural Consensus Theory Midterm Report
By: Una Santos

In this project, I implemented the Cultural Consensus Theory (CCT) using PyMC, as introduced by Romney, Weller, and Batchelder (1986). The goal was to apply the model to a small dataset on plant knowledge, where informants responded to a set of binary questions (1 = correct, 0 = incorrect). I used a combination of 
the professor's starter code and ChatGPT to help structure the model and troubleshoot coding issues. To estimate each informant’s competence (D), I used a Beta distribution with the
parameters alpha = 2 and beta = 1. This assumes informants are more likely than not to be competent but still allows for variability. For each consensus answer (Z), I used a Bernoulli distribution with p = 0.5, which reflects no prior bias toward either answer. These choices helped ensure the model fairly evaluated 
both informants and items. The graphs for the posterior of D and Z are available under data.

The results showed that Informant #6 (D[5]) had the highest competence with a posterior mean of 0.920, while Informant #3 (D[2]) had the lowest at 0.396. Most competence values ranged from 0.4 to 0.9, indicating that the group were generally knowledgeable. All R-hat values were 1.0, suggesting good convergence. The model also estimated the consensus answers (Z) for each question. For example, Z[2] had a mean close to 0.998, indicating very strong agreement on the correct answer for that item. I rounded each Z value to 0 or 1 based on whether its mean was above or below 0.5. Comparing the model's consensus answers to the majority vote showed a 75% match. This difference likely reflects the fact that, unlike the majority vote, the CCT model gives more weight to informants who are estimated to be more competent.


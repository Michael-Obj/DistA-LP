# Response II: two-user running example

The manuscript-ready Section IV insertion is in
[response_ii_running_example.tex](response_ii_running_example.tex).
It uses two neighborhoods of three secrets each and one shared
three-output domain. The numerical coefficient blocks are an
illustrative construction, not measured Rome, NYC, or London results.

## Trace and dimensions

| Stage | User 1 | User 2 | Server |
| --- | --- | --- | --- |
| Private state | True record \(x_2\); \(N_1=\{x_1,x_2,x_3\}\) | True record \(x_5\); \(N_2=\{x_4,x_5,x_6\}\) | Public \(Y=\{y_1,y_2,y_3\}\) |
| Build blocks | \(A_1^{dx}\in\mathbb R^{3\times3}\), \(A_1^{dy},A_1^{cy}\in\mathbb R^{3\times3}\) | Same three shapes | No raw blocks required by the *proposed* protocol |
| Fit and release | Permute locally; fit three surrogate parameter sets; perturb each under its own upload budget | Same | Associate the three released parameter sets using an ephemeral request tag |
| Reconstruct and optimize | — | — | Reconstruct three blocks per request; audit upload-stage loss; run Benders master and two local subproblems using reconstructed coefficients |
| Return and report | Receive \(Z_{N_1\times Y}\), select the row for \(x_2\), sample \(y\in Y\) | Receive \(Z_{N_2\times Y}\), select the row for \(x_5\), sample \(y\in Y\) | Receive only the randomized report in the final stage |

For the toy positions \(x_i=i-1\), \(Y=(0,2,4)\), and absolute
distance, both intra-neighborhood blocks equal
\([0,1,2;1,0,1;2,1,0]\). The two neighborhood-to-output
blocks are \([0,2,4;1,1,3;2,0,2]\) and
\([3,1,1;4,2,0;5,3,1]\). With \(c=d\) and
\(\pi_{x_i}=1/6\), each utility block is its corresponding
neighborhood-to-output block divided by 6. The manuscript
uses symbolic optimized probabilities because no toy LP output
has been measured.

## Mapping to the current MATLAB files

| Protocol step | Existing code | Status in the current experimental driver |
| --- | --- | --- |
| Construct three local blocks | \`Bendersdecomposition/classes/User/User.m\`: \`distance_matrix_LR\`, \`distance_matrix_LR2obf\`, \`cost_matrix_RL\` | Present |
| Fit and perturb SVD parameters | \`User.lowrank_SVD_fit\`, \`User.lowrank_SVD_noisy_parameters\` | Present within \`User.initialization\`; toy rank must be at most 3 |
| Reconstruct three blocks | \`User.lowrank_SVD_recover\` | Present on the user object in the experimental code, before the server is invoked |
| Build WEM split and constraints | \`Subproblem.Q_matrix_cal\`, \`geo_matrix_cal\`, \`unit_matrix_cal\` | Present |
| Solve master/subproblems and add cuts | \`Server.geo_obfuscation_generator\`, \`MasterProgram\`, \`Subproblem\` | Present for the road-network experiment |
| Separate upload interface and anonymous request tag | No corresponding transport or request-tag code located in these classes or \`Bendersdecomposition/main.m\` | Protocol description, not demonstrated by this driver |
| Audited upload mechanism with three separate budget parameters | The driver uses precomputed \`budget_assigned\` and passes \`env_parameters.EPSILON\` into \`User.epsilon\` | The proposed auditor and separate budgeted upload API are not demonstrated in the driver |
| Return block, choose private row, sample categorical output | \`Subproblem.calculate\` assigns \`user.obfuscation_matrix\` | A report-sampling call and separate return channel are not shown |

The driver passes the array of full \`User\` objects into
\`Server.initialization\` and \`Server.geo_obfuscation_generator\`.
Do not use this driver as evidence that only privatized
surrogate parameters are uploaded. The request tag also must
not be described as unlinkable to a later report if both carry
the same tag: it prevents an explicit identity field, but the
server can still associate those messages.

## Checks before changing the paper's privacy claim

- Report the auditor's sampled-pair violation estimate and its
  statistical confidence statement as **average-case**. It does
  not give a bound for each fixed pair.
- Validate the cumulative budget and the relevant failure
  parameters for all three releases and the final report.
- Check constraints against true distances when evaluating
  mDP violations. One-sided parameter noise and metric
  projection do not ensure every reconstructed distance is at
  most its true value.
- Implement and test the separate upload, tag, server
  reconstruction, and report-sampling interfaces before
  presenting them as executed features of the released code.

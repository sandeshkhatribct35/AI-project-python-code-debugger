# Ethics and Limitations

This project includes an Ethics section addressing: privacy, intellectual property, security risks, incorrect AI suggestions, dataset bias, explainability, and human oversight. Key points:

- Do not paste proprietary or secret-bearing code into the app unless you control storage.
- We will avoid sending code to external services by default; document any external calls.
- AI suggestions are recommendations — they may be incorrect and must be reviewed by a human.
- Document dataset provenance and possible biases.
- Use sandboxed or static analysis to reduce execution risks.
 
Dataset provenance and bias
---------------------------

The primary dataset used for initial development is synthetic and contains binary labels only (buggy vs clean). Synthetic data can introduce biases or unrealistic error distributions compared to real-world source code. We will document this provenance, avoid overclaiming real-world performance, and (where possible) augment or validate models on real bug-fix data before presenting final results.

(Full section will be expanded as part of project documentation.)

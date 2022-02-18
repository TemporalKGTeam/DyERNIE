## DyERNIE

Here is the code repository of the EMNLP2020 paper [DyERNIE: Dynamic Evolution of Riemannian Manifold Embeddings for Temporal Knowledge Graph Completion.](https://aclanthology.org/2020.emnlp-main.593.pdf)
#### Installation

- clone repository
- install torch (v1.1.0)  

#### Details of hyperparamters settings please refer to the paper.
#### Run:
Training and validation:
```
python main.py
```
Testing:
```
python main.py --resume_file saved/ckpt_folder_name/checkpoint_X.pt --eval_data test_data --only_eval 
```

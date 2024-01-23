# Reinforcement Learning from Human Feedback (RLHF)

This is a short guide on how to quickly get started training your own RLHF models. Note that this assumes prior access to a machine containing a GPU. All our examples were tested on a machine (`a2-highgpu-1g` on Google Compute Engine) with 12 vCPUs, 85 GB Memory, an NVidia A100 40 GB GPU and 200 GB of disk space.  This comes to about `$1.13` per hour.

For this tutorial, we use Microsoft's DeepSpeed library to train our model, we use OPT-1.3B as our LLM, and we use OPT-350M as our reward model. If you wish to use a different model,  please refer to [this page](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat#-supported-models-) to see models officially supported by DeepSpeed. 

Important Note: The contents of this document were tested a few months ago. As of this writing (October 2nd), DeepSpeed now supports using LLaMa 2 instead. Sample code can be found [here](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/training).

If you use GCP, you can follow this [tutorial](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/generative_ai/rlhf_tune_llm.ipynb) instead.

## Instructions:

1. Install Microsoft DeepSpeed and DeepSpeedExamples:

    ```
    pip install deepspeed>=0.9.0

    git clone https://github.com/microsoft/DeepSpeedExamples.git
    cd DeepSpeedExamples/applications/DeepSpeed-Chat/
    pip install -r requirements.txt
    ```

2. Now we can begin our training. First, supervised fine-tuning (SFT). We use a batch size of 8 instead of the default 16 as our A100 40 GB GPU can't fit higher batch sizes. We also use LoRA for efficient training and gradient accumulation to artificially increase the batch size. We use the four datasets recommended [here](https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/training/README.md). The article also mentions that training for longer results in better performance, even if it does overfit to some extent. With this in mind, we train for 5 epochs instead of just 1. Note that DeepSpeed suggests going as high as 16 epochs.

    ```
    mkdir -p ./output/actor-models/1.3b

    deepspeed --num_gpus 1 training/step1_supervised_finetuning/main.py --model_name_or_path facebook/opt-1.3b --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 8 --lora_dim 128 --zero_stage 0 --data_path Dahoas/rm-static Dahoas/full-hh-rlhf Dahoas/synthetic-instruct-gptj-pairwise yitingxie/rlhf-reward-datasets --num_train_epochs 5 --deepspeed --output_dir ./output/actor-models/1.3b &> ./output/actor-models/1.3b/training.log
    ```

3. Next, we train our reward model (RM). Here we use the default parameters suggested by DeepSpeed, changing only the batch size to 8 so that our model fits in memory.

    ```
    mkdir -p ./output/reward-models/350m


    deepspeed --num_gpus 1 training/step2_reward_model_finetuning/main.py --model_name_or_path facebook/opt-350m --per_device_train_batch_size 8 --per_device_eval_batch_size 8  --num_padding_at_beginning 1 --weight_decay 0.1 --disable_dropout --gradient_accumulation_steps 4 --zero_stage 0 --deepspeed --output_dir ./output/reward-models/350m &> ./output/reward-models/350m/training.log
    ```

4. Finally, we perform step 3. We use the default parameters suggested by DeepSpeed here like before. We change only the batch size to 8.

    ```
    mkdir -p ./output/step3-models/1.3b/

    deepspeed --num_gpus 1 training/step3_rlhf_finetuning/main.py --actor_model_name_or_path ./output/actor-models/1.3b/ --critic_model_name_or_path ./output/reward-models/350m/ --actor_zero_stage 0 --critic_zero_stage 0 --num_padding_at_beginning 1 --gradient_accumulation_steps 2 --deepspeed --actor_lora_dim 128 --enable_hybrid_engine --actor_gradient_checkpointing --disable_actor_dropout --per_device_train_batch_size 8 --per_device_mini_train_batch_size 8 --output_dir ./output/step3-models/1.3b/ &> ./output/step3-models/1.3b/training.log
    ```

5. Now it's time to try out our chatbot! Simply run `inference/chatbot.py` to do that.

    ```
    python inference/chatbot.py --path output/step3-models/actor/
    ```

## Our Results

We build the following table detailing the time taken per model based on our experiments. Note that these numbers are different from the DeepSpeed repository due to different hardware being used.

|      | Step 1 (Epochs) | Step 2 | Step 3 | Total | Cost     |
|------|-----------------|--------|--------|-------|----------|
| 1.3b | ~8.5h (5)       | 1028s  | 5h48m  | ~15h  | ~$17     |
| 6.7b | 5h (2)          | 1028s  | -      | -     | -        |

Time taken per step of training per model. 

We also share the model's metrics at each training step. Note that each step uses different metrics. Step 1 (SFT) uses perplexity, Step 2 (Reward Modeling) uses accuracy & average score and Step 3 uses the average reward score.

|      | Step 1 (Perplexity)  | Step 2 (Average Score) | Step 2 (Accuracy) | Step 3 (Average Reward) |
|------|----------------------|------------------------|-------------------|-------------------------|
| 1.3b | 1.84511              | 9.70320                | 0.63624           | 11.125                  |
| 6.7b | 1.87273              | 9.70320                | 0.63624           | -                       |

**Note**: Since the reward model is independent of the LLM, we used the same reward model in all cases. Thus the metrics and times are the same everywhere.

**Note**: We ran into some issues in running Step3 for OPT-6.7B on our machine and hence the results have not yet been added.

FAQ:

* I'm running into an error containing `symbol cublasLtHSHMatmulAlgoInit version libcublasLt.so` when running code! How do I fix this?

    If you run into this error, the solution is to run `pip uninstall nvidia_cublas_cu11` in your environment. The conflict arises due to torch automatically installing parts of the cuda toolkit which may conflict with your existing installation.

* Can I use a different model instead of the opt models you've used?

Typically, yes. DeepSpeed supports a number of different models, which are listed [here.](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat#-supported-models-) Note that some models may not fit on to a standard GPU out of the box. In such cases you might need to use multi-GPU instances or even multiple nodes. Please refer to the scripts available in the appropriate folder inside the `training_scripts` folder for each step located in `DeepSpeedExamples/applications/DeepSpeed-Chat/training`.

* My chatbot keeps repeating the same few words in its responses. How do I fix this?

    A repeating chatbot seems to be a relatively common issue in smaller models. We've found that passing the `no_repeat_ngram_size=5` parameter to the generation call may fix this issue.

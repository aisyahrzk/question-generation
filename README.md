# T5 Finetuning for Bahasa Melayu Question Generator

This repository contains code and resources for finetuning the T5 (Text-to-Text Transfer Transformer) model to create an answer-agnostic question generator for Bahasa Melayu. The goal of this project is to finetune a T5 model that can generate meaningful and contextually relevant questions for a given text passage in Bahasa Melayu, without relying on specific answers. 


## Dataset

The model is finetune using translated SQUAD bahasa melayu dataset from this [link](https://github.com/huseinzol05/malaysian-dataset/tree/master/question-answer/squad)

The model is able to generate questions without providing the answers. It is trained to generate multiple questions simultaneously by just providing the context. The questions are seperated by the <sep> token. Here's an example of the data the model is trained on:

input text: 
`generate questions: Isaac Newton (1643-1727) mewarisi konsepsi mekanikal Descartes tentang jirim. Dalam ketiga "Rules of Reasoning in Philosophy" beliau, Newton menyenaraikan sifat-sifat sejagat jirim sebagai "sambungan, kekerasan, kebolehpercayaan, mobiliti, dan inersia". Begitu juga dalam Optik dia menyangkal bahawa Tuhan mencipta jirim sebagai "zarah pepejal, besar, keras, tidak dapat ditembusi, boleh bergerak", yang "... walaupun begitu keras sehingga tidak pernah memakai atau memecahkan kepingan". Sifat-sifat "primer" jirim telah dipinda pada keterangan matematik, tidak seperti sifat-sifat "sekunder" seperti warna atau rasa. Seperti Descartes, Newton menolak sifat penting sifat sekunder. </s>`

target text: `Bilakah Descartes dilahirkan? <sep> Apa yang ditulis oleh Descartes? <sep> Apa yang ditolak oleh Newton yang Descartes tidak? <sep> Apa yang dikatakan Descartes adalah sifat-sifat universal jirim? <sep> Kedua-dua sifat primer dan sekunder sesuai dengan bentuk keterangan apa? <sep>	e2e_qg`


## Inference
To generate questions using the model, follow example from question_generator [notebook](question_generator.ipynb).

## Model

Access the model here:[Link](https://huggingface.co/aisyahhrazak/t5-small-bahasa-questiongenerator)


## Reference:

- https://github.com/AMontgomerie/question_generator/tree/master
- Garg, D., Wong, K., Sarangapani, J., & Gupta, S. K. (Eds.). (2021). Advanced Computing. Communications in Computer and Information Science. doi:10.1007/978-981-16-0401-0 
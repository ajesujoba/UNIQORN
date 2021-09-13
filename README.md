# UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text

## Description

Question answering over knowledge graphs and other RDF data has been greatly advanced, with a number of good systems providing crisp answers for natural language questions or telegraphic queries. Some of these systems incorporate textual sources as additional evidence for the answering process, but cannot compute answers that are present in text alone. Conversely, systems from the IR and NLP communities have addressed QA over text, but such systems barely utilize semantic data and knowledge. This paper presents the first QA system that can seamlessly operate over RDF datasets and text corpora, or both together, in a unified framework. Our method, called UNIQORN, builds a context graph on-the-fly, by retrieving question-relevant triples from the RDF data and/or snippets from the text corpus using a fine-tuned BERT model. The resulting graph is typically rich but highly noisy. UNIQORN copes with this input by advanced graph algorithms for Group Steiner Trees, that identify the best answer candidates in the context graph. Experimental results on several benchmarks of complex questions with multiple entities and relations, show that UNIQORN produces results comparable to the state-of-the-art on KGs, text corpora, and heterogeneous sources. The graph-based methodology provides user-interpretable evidence for the complete answering process.

A running example in this paper is:
```Question: director of the western for which Leo won an Oscar? [Answer: Alejandro Iñàrritu]```

<figure>
 <img src="https://user-images.githubusercontent.com/12751379/133112528-08ac04af-e744-4576-bdce-9cb84d0e2096.png" alt="Trulli" style="width:100%">
<!--![xg-kg](https://user-images.githubusercontent.com/12751379/133112528-08ac04af-e744-4576-bdce-9cb84d0e2096.png)-->
<figcaption align = "center"><b>Fig.1 - </b>XG(q) example for KG as input.</figcaption></figure>




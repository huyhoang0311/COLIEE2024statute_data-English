<encoding:UTF-8>

*****  COLIEE-23 Competition in JURISIN 2023 ******

This zip file consists of:
1. Training data 
   train/riteval_H18_en.xml
   train/riteval_H19_en.xml
   train/riteval_H20_en.xml
   train/riteval_H21_en.xml
   train/riteval_H22_en.xml
   train/riteval_H23_en.xml
   train/riteval_H24_en.xml
   train/riteval_H25_en.xml
   train/riteval_H26_en.xml
   train/riteval_H27_en.xml
   train/riteval_H28_en.xml
   train/riteval_H29_en.xml
   train/riteval_H30_en.xml
   train/riteval_H31_en.xml
   train/riteval_R01_en.xml
   train/riteval_R02_en.xml
   train/riteval_R03_en.xml
   train/riteval_R04_en.xml

train/riteval_R04_en.xml is test data used for COLIEE 2023, but due to the problem of the data distributed at the test stage, we excluded results of R4-04-E for the evaluation. However, we modified data of R4-04-E that is good for training data for COLIEE 2024.

2. Japanese Civil Code
   text/civil_code_en-1to724-2.txt

   Remarks for the participants of previous COLIEE:
   Based on the update of Japanese Civil Code at April 2020, we revised text for reflecting this revision for Civil Code and its translation into English. However, since English translated version is not provided for a part of this code, we exclude these parts from the civil code text and questions related to these parts.
   In addition, there are several questions whose labels are modified based on this update. 

3. Example of answers (You should follow this answer format when you submit your results.)
   3.1. For Task 3 (two files are required for submission.)
   	task3.YOURID
	task3-L.YOURID
   3.2 For Task 4 
       	task4.YOURID

4. Format of the Training Data
Training Data is provided in XML format.
The example of the format of the training data is as follows:
-------------------------------
<pair id="H19-1-3" label="N">
<t1>
Article 96
(1) A manifestation of intention based on fraud or duress is voidable.
(2) If a third party commits a fraud inducing a first party to make a manifestation of intention to a second party, that manifestation of intention is voidable only if the second party knew or could have known that fact.
(3) The rescission of a manifestation of intention induced by fraud under the provisions of the preceding two paragraphs may not be duly asserted against a third party in good faith acting without negligence..
</t1>
<t2>
A person who made a manifestation of intention which was induced by duress emanated from a third party may rescind such manifestation of intention on the basis of duress, only if the other party knew or was negligent of such fact.
</t2>
</pair>
-------------------------------
The tag <t2> is a query sentence and the tag <t1> shows corresponding civil codes. 

4.1 Answer for Task 3
The purpose of the Task 3 is to find corresponding articles of a given query sentence. In the example "H19-1-3" above, if the relevant civil code in your system is "96", and the relevance score is 0.84, then the answer for the Task 3 that you submit will be

H19-1-3 Q0 96 1 0.84 YOURID

where YOURID should be replaced with your group id.  The first token of each line (whitespace delimited) is the question ID, the second column is always 'Q0' according to the trec_eval program. The third token is the corresponding civil code that you obtained, the fourth one is the rank, the fifth token is the similarity score (floating value), and the last token is your ID. Each line can include only one civil code. If you obtained multiple civil codes as relevant, you have to add more lines as follows with their ranks and similarity scores:

H19-1-3 Q0 96 1 0.84 YOURID
H19-1-3 Q0 128 2 0.73 YOURID

In the test data for the Task 3, the <t1> and label information will not be given. Only <t2> information which is a query sentence will be given.

4.2 Answer for Task 4
The purpose of the Task 4 is to answer Yes/No combining your information retrieval (IR) system (Task 3) and your entailment system.
First, you retrieve relevant articles using your IR system given a query, and then you induce 'yes' or 'no' using your entailment system between the query sentence and your retrieved articles.

In the example of "H19-1-3", based on the entailment from the corresponding article 96, the answer of the query sentence <t1> is "No", and you can also validate your Yes/No answer using the "label" information. If the label is "Y", it means the answer of the query sentence is "Yes". Otherwise, if the label is "N", it means the answer of the query sentence is "No".
In this example, the answer for the Task 4 is as following:

H19-1-3 N YOURID

The test data of Task 4 will include only <t2> information which is a query. The <t1> and label information will not be given.

Please see https://sites.ualberta.ca/~rabelo/COLIEE2024/ for details. If you have any queries, please do not hesitate to contact us at yoshioka@ist.hokudai.ac.jp . Thank you for the participation of COLIEE-2024 and we look forward to your result submission.

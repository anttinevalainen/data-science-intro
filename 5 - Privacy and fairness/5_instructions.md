# Week 5 - Privacy and fairness

## Exercise 1 - Privacy and Data Protection

First, look up the European Data Protection Regulation (“GDPR”). We will refer to the articles and their parts by, e.g., “Art 6 (1) a) GDPR” which means Article 6 (“Lawfulness of processing”), first paragraph, item a in the GDPR.

### 1. Valid Consent?

Find a service you use to which you have given consent for the processing of your personal data (Art 6 (1) a) GDPR). Have a look at the privacy notices, policies, or settings of this service.

Are the basic legal conditions for this consent in your opinion in line with the new requirements and conditions set by the GDPR? You should provide an answer with justification based on the GDPR, where you refer to specific articles and paragraphs.

```bash
I am a long time user of Discogs, which is an online community and marketplace for music
albums, and everything related. In Discogs, I have saved my personal data for quicker
checkout during purchases. This data includes my full name and my address details. The
payments are done via PayPal, so Discogs does not store my billing information.
-
When reading the privacy policy of Discogs, the service has listed the same lawful bases,
that are stated on the GDPR Art 6(1). This states the manner and reason for the data
collecting and justifies the collection of users’ data on the same principles.
```

### 2. Your Right to Access your Personal Data

You have the right to know if personal data about you is processed by a controller. You also have the right to get access to, for example, the processing purposes, the data categories, data transfers, and duration of storage.

Find the relevant parts in GDPR and study your rights as a “data subject”. File a right to access -request with a data processing service of your choosing. Describe the mechanism that is put in place by the service to enable you to exercise this right (if any).

Whether you get a response or not, think about how well your rights as a data subject are respected in practice. Your answer should again refer to specific articles and paragraphs of the GDPR.

```bash
I filed a right to access -request to Discogs via their website, where the request could be
done easily. The request goes as the following:
-
’’As a client, I have the right to know when personal data about myself is processed by a
controller. Therefore, I also have the right to get access to said information.
As stated in the GDPR Art 15 (1(a-d)) (file), I request the information about the processing
purposes (Art 15(1(a)), the data categories (Art 15(1(b)), data transfers (Art 15(1(c)), and
duration of storage of data (Art 15(1(d)) concerning data about me.’’
-
The requesting via website tool rather than email seems simple and an easy approach to
the matter. I did believe that I would receive a respectful answer to the manner and the service
does hold on to their privacy policies. I received an answer two days later, which contained
all the information that has been stated to be collected in the privacy policy of the service.
Each piece of information gathered had been continued with the statement according to
each line in the Art 15 (1(a-d).
-
As stated in the Art 15(1), the same information is also mentioned in the privacy policy of
the Discogs website.
```

### 3. Anonymisation & Pseudonymisation

What is the difference between anonymisation and pseudonymisation of personal data?

```bash
When data is being anonymized, any tracks of pointing out the data to a specific person is
erased or altered in a way that the data becomes anonymous and it is not possible to
connect to a specific person.
-
When data is being pseudonymized, above manners are done, but the data is possible to be connected
to specific people/person with additional data, which is not openly available/is only in use of the
provider and the subject themselves.
```

## Exercise 2 - Fairness-aware AI
In this exercise, you should simulate three different discrimination scenarios (direct, indirect and non-discriminative) in a simple linear regression setting.

Create a template that generates data about working hours and salaries of `n = 5000` people. The salary equals `100 x working hours` plus/minus normal distributed noise. Running the code should produce a scatter plot with `gender = 0` (men) in one color and `gender = 1` (women) in other color. Add a trend line for each group to the plot, and an overall trend line for all data combined. A linear regression model learned from the data without the protected characteristic (gender) should have slope close to `100.0`

Now, edit the code to simulate the following scenarios:

a) The salary of women is reduced by `200` euros ("direct discrimination")

b) The working hours of men are binomially distributed with parameters `(60, 0.55)` while the working hours of women are binomially distributed with parameters `(60, 0.45)` ("no discrimination")

c) Both of the above changes at the same time ("indirect discrimination")

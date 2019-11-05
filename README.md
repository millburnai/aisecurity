# AI Security

## Overview

In the past few years, countless mass shootings and violent attacks have created a social climate of fear. Shocking crimes like school shootings have become the norm across the nation. Though our country’s leaders are scrambling for a solution, large-scale change cannot happen overnight. In the meantime, it is our community’s responsibility to do its best to avoid these atrocities.

We propose AISecurity, a machine-learning-driven addition to Millburn’s security system. AISecurity will use facial recognition in order to better monitor the flow of people in and out of the school. It will be built by the students of the AI program, a part of the Computer Science Integrated Initiative.

More specifically, AISecurity will be able to detect who is coming and going through the main doors. Depending on the wants of the administration, AISecurity can be used in several ways, such as logging who’s leaving or entering school or enhancing the buzz-in or kiosk system. Designed specifically to minimize privacy concerns, AISecurity will not store the pictures of any people it sees. AISecurity is not a program to monitor students’ every move; rather, it is a supplement that will simply help keep track of the flow of people in and out of the school. It is designed with both the school’s safety and privacy in mind: everyone wants better security, but nobody wants a camera following their every move.

Ultimately, our goal is twofold: enhance security while simultaneously providing a learning opportunity for students in the AI program. Inspired by Millburn High School’s student leaders, AISecurity is intended to be a big step forward in terms of student-led community initiatives. Through this project, we’ll learn invaluable real-world skills while creating a novel and privacy-conscious way to better our school environment.


## Privacy

In addition to Millburn’s cybersecurity system, AISecurity has an internal three-point security defense to ensure that any personal data is inaccessible by humans.

1. __Database processing__: Instead of storing actual images, AISecurity will use a mathematical algorithm to convert pictures into a format that is not recognizable by humans or machines. This transformation cannot be reversed: it is not possible to retrieve the original image from the processed result.
2. __Encryption__: As another security measure, AISecurity will take the processed database and apply AES encryption to it. AES is a NIST-approved and currently unbroken encryption algorithm used by the US government since 2002 for top classified information.
3. __Locality__: All this information is stored locally on the school server. As a result, it would be difficult anyone to access these images without permission.

The combination of these three security measures means that nobody will be able to view the student or faculty images. There are no exceptions to this rule: it is not mathematically possible to view the original image of anybody in the database. Additionally, during the use of AISecurity, none of the images used for recognition will be stored, ensuring maximum privacy. In reality, there should be little to no privacy concerns, since the reference database is impossible to access and real-time images are not stored.


## Logistics

AISecurity will be created by a select team from the members of the CSII AI program. We estimate three to four months to create and finalize this project, and hope to have a finished project by May.

In order to make the implementation of AISecurity as easy as possible, we propose using the existing camera infrastructure. AISecurity will be designed to run on any camera: given video feed, AISecurity will be able to run successfully using that data. For example, the doorbell camera currently used for the buzz-in system can be simultaneously used for AISecurity.

As for the facial recognition algorithm, AISecurity will employ a breakthrough 2015 method designed by Google AI researchers and presented in the paper “FaceNet: A Unified Embedding for Face Recognition and Clustering”.

All information recorded through this camera will be logged in a text file. When a person is recognized, the log will automatically update with the timestamp and the person’s name-- the image of the person is not stored, both for privacy and storage concerns. If a person is not recognized, their presence and timestamp is logged accordingly. This log will be stored and encrypted similarly to the people database.


## Usage

As a demonstration of its capabilities, AISecurity will be first integrated with the kiosk project. A mini-computer called the NVIDIA Jetson Nano will replace the Raspberry Pi in one of the kiosks, providing the necessary computing power to run facial recognition on the kiosk. Integration should be relatively easy because the Jetson Nano and Pi are similar, meaning a swap should be fairly painless.

After successfully completing this demonstration, AISecurity can be moved to other applications if desired.

DROP DATABASE IF EXISTS LOG
CREATE DATABASE LOG
USE LOG
CREATE TABLE Transactions (transid INTEGER(5), studentid INTEGER(5), studentname VARCHAR (200), date_ DATE, time_ TIME)
CREATE TABLE Suspicious (path_to_img VARCHAR(200), date_ DATE, time_ TIME)
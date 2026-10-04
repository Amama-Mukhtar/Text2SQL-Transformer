# Qualitative samples (dev, beam)

Values are lower-cased because the model works on lower-cased text.

## Correct

### 1
**Question:** How many schools did player number 3 play at?

- Gold: `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
- Ours (beam): `SELECT COUNT(School/Club Team) FROM table WHERE No. = '3'`
- Verdict: correct

### 2
**Question:** What school did player number 21 play for?

- Gold: `SELECT School/Club Team FROM table WHERE No. = '21'`
- Ours (beam): `SELECT School/Club Team FROM table WHERE No. = '21'`
- Verdict: correct

### 3
**Question:** Who is the player that wears number 42?

- Gold: `SELECT Player FROM table WHERE No. = '42'`
- Ours (beam): `SELECT Player FROM table WHERE No. = '42'`
- Verdict: correct

### 4
**Question:** What player played guard for toronto in 1996-97?

- Gold: `SELECT Player FROM table WHERE Position = 'Guard' AND Years in Toronto = '1996-97'`
- Ours (beam): `SELECT Player FROM table WHERE Position = 'guard' AND Years in Toronto = '1996-97'`
- Verdict: correct

### 5
**Question:** Who are all of the players on the Westchester High School club team?

- Gold: `SELECT Player FROM table WHERE School/Club Team = 'Westchester High School'`
- Ours (beam): `SELECT Player FROM table WHERE School/Club Team = 'westchester high school'`
- Verdict: correct

## Wrong

### 1
**Question:** What position does the player who played for butler cc (ks) play?

- Gold: `SELECT Position FROM table WHERE School/Club Team = 'Butler CC (KS)'`
- Ours (beam): `SELECT Position FROM table WHERE Player = 'butler cc (ks)'`
- Verdict: wrong condition column

### 2
**Question:** What is the English name of the country whose official native language is Dutch Papiamento?

- Gold: `SELECT Country ( exonym ) FROM table WHERE Official or native language(s) (alphabet/script) = 'Dutch Papiamento'`
- Ours (beam): `SELECT Official or native language(s) (alphabet/script) FROM table WHERE Country ( exonym ) = 'dutch papiamento'`
- Verdict: wrong select column

### 3
**Question:** Name the minimum tiesplayed for 6 years

- Gold: `SELECT MIN(Ties played) FROM table WHERE Years played = '6'`
- Ours (beam): `SELECT MIN(Ties played) FROM table WHERE Years played = '6 years'`
- Verdict: wrong value or operator

### 4
**Question:** how many division  did not qualify for u.s. open cup in 2003

- Gold: `SELECT Division FROM table WHERE U.S. Open Cup = 'Did Not Qualify' AND Year = '2003'`
- Ours (beam): `SELECT COUNT(Division) FROM table WHERE U.S. Open Cup = 'not qualify' AND Year = '2003'`
- Verdict: wrong aggregation

### 5
**Question:** Provide with the names of the village (German) that is part of village (Slovenian) with sele srednji kot.

- Gold: `SELECT Village (German) FROM table WHERE Village (Slovenian) = 'Sele Srednji Kot'`
- Ours (beam): `SELECT Village (German) FROM table WHERE Village (Slovenian) = 'village (slovenian)' AND Number of people 1991 = 'kot'`
- Verdict: extra condition


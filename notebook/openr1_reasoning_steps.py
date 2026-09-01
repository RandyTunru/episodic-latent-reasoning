{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "50db6993",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from dotenv import load_dotenv\n",
    "load_dotenv()  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0a0b54f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "data = load_dataset(\"open-r1/Mixture-of-Thoughts\", \"science\", split=\"train\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1e8c9118",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>messages</th>\n",
       "      <th>num_tokens</th>\n",
       "      <th>source</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>[{'content': 'What are examples of food-safe p...</td>\n",
       "      <td>2039</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>[{'content': 'What are examples of food-safe p...</td>\n",
       "      <td>2139</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>[{'content': 'What are examples of food-safe p...</td>\n",
       "      <td>2661</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>[{'content': 'What is the hybridization of the...</td>\n",
       "      <td>917</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>[{'content': 'What is the hybridization of the...</td>\n",
       "      <td>786</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172509</th>\n",
       "      <td>[{'content': 'How can you determine if the occ...</td>\n",
       "      <td>6015</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172510</th>\n",
       "      <td>[{'content': 'What is the difference in molecu...</td>\n",
       "      <td>2157</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172511</th>\n",
       "      <td>[{'content': 'What is the eccentricity of the ...</td>\n",
       "      <td>7834</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172512</th>\n",
       "      <td>[{'content': 'How many one-loop 1PI graphs are...</td>\n",
       "      <td>6223</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>172513</th>\n",
       "      <td>[{'content': 'How many independent properties ...</td>\n",
       "      <td>6505</td>\n",
       "      <td>nvidia/Llama-Nemotron-Post-Training-Dataset</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>172514 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                 messages  num_tokens  \\\n",
       "0       [{'content': 'What are examples of food-safe p...        2039   \n",
       "1       [{'content': 'What are examples of food-safe p...        2139   \n",
       "2       [{'content': 'What are examples of food-safe p...        2661   \n",
       "3       [{'content': 'What is the hybridization of the...         917   \n",
       "4       [{'content': 'What is the hybridization of the...         786   \n",
       "...                                                   ...         ...   \n",
       "172509  [{'content': 'How can you determine if the occ...        6015   \n",
       "172510  [{'content': 'What is the difference in molecu...        2157   \n",
       "172511  [{'content': 'What is the eccentricity of the ...        7834   \n",
       "172512  [{'content': 'How many one-loop 1PI graphs are...        6223   \n",
       "172513  [{'content': 'How many independent properties ...        6505   \n",
       "\n",
       "                                             source  \n",
       "0       nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "1       nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "2       nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "3       nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "4       nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "...                                             ...  \n",
       "172509  nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "172510  nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "172511  nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "172512  nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "172513  nvidia/Llama-Nemotron-Post-Training-Dataset  \n",
       "\n",
       "[172514 rows x 3 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e381da03",
   "metadata": {},
   "outputs": [],
   "source": [
    "start_tag = \"<think>\"\n",
    "end_tag = \"</think>\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1317c45d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[{'content': 'What are examples of food-safe polyelectrolytes that act as bases and develop positive charges along the polymer?A: glucosamine, chitin, chitosan\\nB: xylan, mannan, galactan\\nC: pectin, alginate, carrageenan\\nD: cellulose, lignin, hemi cellulose',\n",
       "  'role': 'user'},\n",
       " {'content': '<think>\\nOkay, let me try to figure this out. The question is asking about food-safe polyelectrolytes that act as bases and develop positive charges along the polymer. The options are A to D, each listing different polysaccharides or related compounds. \\n\\nFirst, I need to recall what polyelectrolytes are. They are polymers that have charged groups, right? So, they can be either polycations (positively charged) or polyanions (negatively charged) when dissolved in water. The question specifically mentions bases, which typically accept protons (H+), so maybe these polymers would become protonated, leading to a positive charge. \\n\\nNow, looking at the options. Let\\'s go through each:\\n\\nOption A: glucosamine, chitin, chitosan. Wait, glucosamine is a monosaccharide derivative, an amino sugar. Chitin is a polymer of N-acetylglucosamine, and chitosan is deacetylated chitin, which means it has more amino groups. Amino groups can act as weak bases—when they are protonated, they become positively charged (NH3+). So chitosan is known to be a cationic polysaccharide. But what about chitin? If it\\'s N-acetylated, the amino groups are acetylated, so they might not protonate as easily. So maybe chitosan is the main one here. Glucosamine itself is a monomer, so maybe they are including it as a component?\\n\\nOption B: xylan, mannan, galactan. Those are all polysaccharides. Xylan is a hemicellulose with xylose units; mannan has mannose, galactan has galactose. Do these have charged groups? They might be neutral or have some substitutions. For example, some xylans might have uronic acid groups, but those would make them polyanions. Not sure if these are bases. If they aren\\'t charged, then perhaps they are neutral, which wouldn\\'t fit the polyelectrolyte part. So maybe this option isn\\'t right.\\n\\nOption C: pectin, alginate, carrageenan. I know pectin has lots of galacturonic acid (which is a carboxylic acid group), alginate has guluronic and mannuronic acids, and carrageenan has sulfate groups. All of these are acidic groups that would lose protons and become negatively charged. So these would be polyanions, acting as acids. So this option is probably the opposite of what the question is asking for. So maybe C is not correct.\\n\\nOption D: cellulose, lignin, hemicellulose. Cellulose is a linear polymer of glucose, which doesn\\'t have charged groups. Lignin is a complex polyphenolic polymer, not a polysaccharide, and while it might have some charged groups depending on pH, hemicellulose (a mix like xylan or mannan?) again, not sure. I don\\'t think these are polyelectrolytes. Cellulose is neutral, hemicellulose may have some substituents but again, perhaps not as a charged polymer. Lignin probably doesn\\'t act as a base here. So D might not be correct.\\n\\nSo coming back. The question wants polymers that act as bases (proton acceptors) and thus have positive charges. So amino groups when protonated (NH3+). So chitosan (from option A) is known for being positively charged in acidic conditions. Chitin is less so because of acetylation. Glucosamine as a monomer, but maybe in the context of food, all three in option A are considered. Wait, but chitin is also a polysaccharide. Let me check: chitin is made of N-acetylglucosamine units linked. So the amino groups are acetylated. So in such cases, chitin\\'s amino groups would be blocked by acetyl groups, so they can\\'t protonate. Therefore, chitin is not a polycation. So chitosan is deacetylated, so more amino groups are free to accept protons and become positive.\\n\\nSo only chitosan in option A is a base. But what about the others? Glucosamine isn\\'t a polymer, right? Wait, the options are labeled as \"polyelectrolytes\" which are polymers. So if glucosamine is a monomer, can that count? Maybe the question lists glucosamine as part of the answer. Wait the options just list three substances; perhaps in the list, two are polymers. Like maybe glucosamine is included as part of chitosan or chitin? Hmm. Maybe I need to check this again.\\n\\nAlternatively, perhaps there\\'s confusion here. But in food applications, chitosan is the main one. For example, in edible coatings, chitosan is used because it can form films and has antimicrobial properties due to its positive charge. Chitin itself is less soluble in water unless processed. So option A\\'s answer would be glucosamine (monomer), chitin (polymer with N-acetyl), and chitosan (deacetylated). So in terms of charged, chitosan would be the cationic one. But the question states that all three in option A are examples. So even if chitin and chitosan are part of them, but what about glucosamine? Could that also be a base in some form? Wait, but as a monomer, glucosamine has an amino group which can protonate. If it\\'s a polymer, then ... Well, but the question says polyelectrolytes, which are polymers. So glucosamine as a monomer may not be. Unless there\\'s a polymer made from glucosamine. Wait, but normally, chitin is made from N-acetylglucosamine, chitosan from glucosamine (after deacetylation). So, the glucosamine listed here in option A could be misleading because it\\'s the monomer, while chitin and chitosan are polymers. Hmm, but maybe the question allows for including all three. But since the answer options have three, and if two of the three are correct, but one is not. If that\\'s the case, maybe the answer is A.\\n\\nLooking at the other options again. Option C\\'s examples are all known polyanions. B\\'s are neutral. D\\'s are mostly neutral. So the only options with cationic (basic) polyelectrolytes would be A. Even though one of them (glucosamine) isn\\'t a polymer, maybe that\\'s an error in the options, but among the given choices, A is the best fit. So the answer is A.\\n</think>\\n\\nTo determine the correct answer, we analyze the characteristics of each option based on the requirement for food-safe polyelectrolytes that act as bases and develop positive charges:\\n\\n1. **Option A: Glucosamine, Chitin, Chitosan**  \\n   - **Chitosan** is a deacetylated derivative of chitin, containing free amino groups that protonate in acidic conditions, forming positively charged ammonium ions (NH₃⁺). This makes it a well-known cationic polyelectrolyte used in food applications.  \\n   - **Chitin**, the precursor to chitosan, has N-acetylglucosamine units. The acetylated amino groups in chitin cannot protonate easily, so it is not typically cationic.  \\n   - **Glucosamine** is a monomeric amino sugar, not a polymer, and thus not a polyelectrolyte. Despite this inconsistency, chitosan is the key example here, making Option A the most plausible choice.  \\n\\n2. **Option B: Xylan, Mannan, Galactan**  \\n   - These are neutral or weakly acidic polysaccharides (e.g., xylan may have uronic acid groups). They do not act as bases or develop positive charges, ruling out this option.  \\n\\n3. **Option C: Pectin, Alginate, Carrageenan**  \\n   - These are all polyanions. Pectin (galacturonic acid), alginate (guluronic/mannuronic acids), and carrageenan (sulfate groups) release protons to become negatively charged. They are acidic, not basic, and thus incorrect.  \\n\\n4. **Option D: Cellulose, Lignin, Hemicellulose**  \\n   - Cellulose and hemicellulose are neutral polysaccharides. Lignin is a polyphenolic polymer with no inherent cationic properties. None of these act as bases or develop positive charges.  \\n\\nWhile Option A includes the non-polymeric glucosamine and the less-charged chitin, **chitosan** is the primary food-safe cationic polyelectrolyte in the list. Given the structure of the question, **A is the best answer** despite minor inconsistencies. \\n\\n\\\\boxed{A}',\n",
       "  'role': 'assistant'}]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_0 = data['messages'][0]\n",
    "sample_0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "67e0e0f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "instruction = sample_0[0]['content']\n",
    "assistant_response = sample_0[1]['content']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4c575e52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'What are examples of food-safe polyelectrolytes that act as bases and develop positive charges along the polymer?A: glucosamine, chitin, chitosan\\nB: xylan, mannan, galactan\\nC: pectin, alginate, carrageenan\\nD: cellulose, lignin, hemi cellulose'"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "instruction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "57558058",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'To determine the correct answer, we analyze the characteristics of each option based on the requirement for food-safe polyelectrolytes that act as bases and develop positive charges:\\n\\n1. **Option A: Glucosamine, Chitin, Chitosan**  \\n   - **Chitosan** is a deacetylated derivative of chitin, containing free amino groups that protonate in acidic conditions, forming positively charged ammonium ions (NH₃⁺). This makes it a well-known cationic polyelectrolyte used in food applications.  \\n   - **Chitin**, the precursor to chitosan, has N-acetylglucosamine units. The acetylated amino groups in chitin cannot protonate easily, so it is not typically cationic.  \\n   - **Glucosamine** is a monomeric amino sugar, not a polymer, and thus not a polyelectrolyte. Despite this inconsistency, chitosan is the key example here, making Option A the most plausible choice.  \\n\\n2. **Option B: Xylan, Mannan, Galactan**  \\n   - These are neutral or weakly acidic polysaccharides (e.g., xylan may have uronic acid groups). They do not act as bases or develop positive charges, ruling out this option.  \\n\\n3. **Option C: Pectin, Alginate, Carrageenan**  \\n   - These are all polyanions. Pectin (galacturonic acid), alginate (guluronic/mannuronic acids), and carrageenan (sulfate groups) release protons to become negatively charged. They are acidic, not basic, and thus incorrect.  \\n\\n4. **Option D: Cellulose, Lignin, Hemicellulose**  \\n   - Cellulose and hemicellulose are neutral polysaccharides. Lignin is a polyphenolic polymer with no inherent cationic properties. None of these act as bases or develop positive charges.  \\n\\nWhile Option A includes the non-polymeric glucosamine and the less-charged chitin, **chitosan** is the primary food-safe cationic polyelectrolyte in the list. Given the structure of the question, **A is the best answer** despite minor inconsistencies. \\n\\n\\\\boxed{A}'"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "generated_message = assistant_response.split(end_tag)[1].strip()\n",
    "generated_message"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6ff2c86f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Okay, let me try to figure this out. The question is asking about food-safe polyelectrolytes that act as bases and develop positive charges along the polymer. The options are A to D, each listing different polysaccharides or related compounds. \\n\\nFirst, I need to recall what polyelectrolytes are. They are polymers that have charged groups, right? So, they can be either polycations (positively charged) or polyanions (negatively charged) when dissolved in water. The question specifically mentions bases, which typically accept protons (H+), so maybe these polymers would become protonated, leading to a positive charge. \\n\\nNow, looking at the options. Let\\'s go through each:\\n\\nOption A: glucosamine, chitin, chitosan. Wait, glucosamine is a monosaccharide derivative, an amino sugar. Chitin is a polymer of N-acetylglucosamine, and chitosan is deacetylated chitin, which means it has more amino groups. Amino groups can act as weak bases—when they are protonated, they become positively charged (NH3+). So chitosan is known to be a cationic polysaccharide. But what about chitin? If it\\'s N-acetylated, the amino groups are acetylated, so they might not protonate as easily. So maybe chitosan is the main one here. Glucosamine itself is a monomer, so maybe they are including it as a component?\\n\\nOption B: xylan, mannan, galactan. Those are all polysaccharides. Xylan is a hemicellulose with xylose units; mannan has mannose, galactan has galactose. Do these have charged groups? They might be neutral or have some substitutions. For example, some xylans might have uronic acid groups, but those would make them polyanions. Not sure if these are bases. If they aren\\'t charged, then perhaps they are neutral, which wouldn\\'t fit the polyelectrolyte part. So maybe this option isn\\'t right.\\n\\nOption C: pectin, alginate, carrageenan. I know pectin has lots of galacturonic acid (which is a carboxylic acid group), alginate has guluronic and mannuronic acids, and carrageenan has sulfate groups. All of these are acidic groups that would lose protons and become negatively charged. So these would be polyanions, acting as acids. So this option is probably the opposite of what the question is asking for. So maybe C is not correct.\\n\\nOption D: cellulose, lignin, hemicellulose. Cellulose is a linear polymer of glucose, which doesn\\'t have charged groups. Lignin is a complex polyphenolic polymer, not a polysaccharide, and while it might have some charged groups depending on pH, hemicellulose (a mix like xylan or mannan?) again, not sure. I don\\'t think these are polyelectrolytes. Cellulose is neutral, hemicellulose may have some substituents but again, perhaps not as a charged polymer. Lignin probably doesn\\'t act as a base here. So D might not be correct.\\n\\nSo coming back. The question wants polymers that act as bases (proton acceptors) and thus have positive charges. So amino groups when protonated (NH3+). So chitosan (from option A) is known for being positively charged in acidic conditions. Chitin is less so because of acetylation. Glucosamine as a monomer, but maybe in the context of food, all three in option A are considered. Wait, but chitin is also a polysaccharide. Let me check: chitin is made of N-acetylglucosamine units linked. So the amino groups are acetylated. So in such cases, chitin\\'s amino groups would be blocked by acetyl groups, so they can\\'t protonate. Therefore, chitin is not a polycation. So chitosan is deacetylated, so more amino groups are free to accept protons and become positive.\\n\\nSo only chitosan in option A is a base. But what about the others? Glucosamine isn\\'t a polymer, right? Wait, the options are labeled as \"polyelectrolytes\" which are polymers. So if glucosamine is a monomer, can that count? Maybe the question lists glucosamine as part of the answer. Wait the options just list three substances; perhaps in the list, two are polymers. Like maybe glucosamine is included as part of chitosan or chitin? Hmm. Maybe I need to check this again.\\n\\nAlternatively, perhaps there\\'s confusion here. But in food applications, chitosan is the main one. For example, in edible coatings, chitosan is used because it can form films and has antimicrobial properties due to its positive charge. Chitin itself is less soluble in water unless processed. So option A\\'s answer would be glucosamine (monomer), chitin (polymer with N-acetyl), and chitosan (deacetylated). So in terms of charged, chitosan would be the cationic one. But the question states that all three in option A are examples. So even if chitin and chitosan are part of them, but what about glucosamine? Could that also be a base in some form? Wait, but as a monomer, glucosamine has an amino group which can protonate. If it\\'s a polymer, then ... Well, but the question says polyelectrolytes, which are polymers. So glucosamine as a monomer may not be. Unless there\\'s a polymer made from glucosamine. Wait, but normally, chitin is made from N-acetylglucosamine, chitosan from glucosamine (after deacetylation). So, the glucosamine listed here in option A could be misleading because it\\'s the monomer, while chitin and chitosan are polymers. Hmm, but maybe the question allows for including all three. But since the answer options have three, and if two of the three are correct, but one is not. If that\\'s the case, maybe the answer is A.\\n\\nLooking at the other options again. Option C\\'s examples are all known polyanions. B\\'s are neutral. D\\'s are mostly neutral. So the only options with cationic (basic) polyelectrolytes would be A. Even though one of them (glucosamine) isn\\'t a polymer, maybe that\\'s an error in the options, but among the given choices, A is the best fit. So the answer is A.'"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reasoning_text = assistant_response.split(start_tag)[1].split(end_tag)[0]\n",
    "reasoning_text.strip()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a0a582f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Okay, let me try to figure this out. The question is asking about food-safe polyelectrolytes that act as bases and develop positive charges along the polymer. The options are A to D, each listing different polysaccharides or related compounds.',\n",
       " 'First, I need to recall what polyelectrolytes are. They are polymers that have charged groups, right? So, they can be either polycations (positively charged) or polyanions (negatively charged) when dissolved in water. The question specifically mentions bases, which typically accept protons (H+), so maybe these polymers would become protonated, leading to a positive charge.',\n",
       " \"Now, looking at the options. Let's go through each:\",\n",
       " \"Option A: glucosamine, chitin, chitosan. Wait, glucosamine is a monosaccharide derivative, an amino sugar. Chitin is a polymer of N-acetylglucosamine, and chitosan is deacetylated chitin, which means it has more amino groups. Amino groups can act as weak bases—when they are protonated, they become positively charged (NH3+). So chitosan is known to be a cationic polysaccharide. But what about chitin? If it's N-acetylated, the amino groups are acetylated, so they might not protonate as easily. So maybe chitosan is the main one here. Glucosamine itself is a monomer, so maybe they are including it as a component?\",\n",
       " \"Option B: xylan, mannan, galactan. Those are all polysaccharides. Xylan is a hemicellulose with xylose units; mannan has mannose, galactan has galactose. Do these have charged groups? They might be neutral or have some substitutions. For example, some xylans might have uronic acid groups, but those would make them polyanions. Not sure if these are bases. If they aren't charged, then perhaps they are neutral, which wouldn't fit the polyelectrolyte part. So maybe this option isn't right.\",\n",
       " 'Option C: pectin, alginate, carrageenan. I know pectin has lots of galacturonic acid (which is a carboxylic acid group), alginate has guluronic and mannuronic acids, and carrageenan has sulfate groups. All of these are acidic groups that would lose protons and become negatively charged. So these would be polyanions, acting as acids. So this option is probably the opposite of what the question is asking for. So maybe C is not correct.',\n",
       " \"Option D: cellulose, lignin, hemicellulose. Cellulose is a linear polymer of glucose, which doesn't have charged groups. Lignin is a complex polyphenolic polymer, not a polysaccharide, and while it might have some charged groups depending on pH, hemicellulose (a mix like xylan or mannan?) again, not sure. I don't think these are polyelectrolytes. Cellulose is neutral, hemicellulose may have some substituents but again, perhaps not as a charged polymer. Lignin probably doesn't act as a base here. So D might not be correct.\",\n",
       " \"So coming back. The question wants polymers that act as bases (proton acceptors) and thus have positive charges. So amino groups when protonated (NH3+). So chitosan (from option A) is known for being positively charged in acidic conditions. Chitin is less so because of acetylation. Glucosamine as a monomer, but maybe in the context of food, all three in option A are considered. Wait, but chitin is also a polysaccharide. Let me check: chitin is made of N-acetylglucosamine units linked. So the amino groups are acetylated. So in such cases, chitin's amino groups would be blocked by acetyl groups, so they can't protonate. Therefore, chitin is not a polycation. So chitosan is deacetylated, so more amino groups are free to accept protons and become positive.\",\n",
       " 'So only chitosan in option A is a base. But what about the others? Glucosamine isn\\'t a polymer, right? Wait, the options are labeled as \"polyelectrolytes\" which are polymers. So if glucosamine is a monomer, can that count? Maybe the question lists glucosamine as part of the answer. Wait the options just list three substances; perhaps in the list, two are polymers. Like maybe glucosamine is included as part of chitosan or chitin? Hmm. Maybe I need to check this again.',\n",
       " \"Alternatively, perhaps there's confusion here. But in food applications, chitosan is the main one. For example, in edible coatings, chitosan is used because it can form films and has antimicrobial properties due to its positive charge. Chitin itself is less soluble in water unless processed. So option A's answer would be glucosamine (monomer), chitin (polymer with N-acetyl), and chitosan (deacetylated). So in terms of charged, chitosan would be the cationic one. But the question states that all three in option A are examples. So even if chitin and chitosan are part of them, but what about glucosamine? Could that also be a base in some form? Wait, but as a monomer, glucosamine has an amino group which can protonate. If it's a polymer, then ... Well, but the question says polyelectrolytes, which are polymers. So glucosamine as a monomer may not be. Unless there's a polymer made from glucosamine. Wait, but normally, chitin is made from N-acetylglucosamine, chitosan from glucosamine (after deacetylation). So, the glucosamine listed here in option A could be misleading because it's the monomer, while chitin and chitosan are polymers. Hmm, but maybe the question allows for including all three. But since the answer options have three, and if two of the three are correct, but one is not. If that's the case, maybe the answer is A.\",\n",
       " \"Looking at the other options again. Option C's examples are all known polyanions. B's are neutral. D's are mostly neutral. So the only options with cationic (basic) polyelectrolytes would be A. Even though one of them (glucosamine) isn't a polymer, maybe that's an error in the options, but among the given choices, A is the best fit. So the answer is A.\"]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reasoning_steps = [step.strip() for step in reasoning_text.split('\\n\\n') if step.strip()]\n",
    "reasoning_steps"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pytorch-training",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

import sqlite3
conn = sqlite3.connect("Diseases.db")
c = conn.cursor()

diseases_data = [
    ('EB001','Early Blight','This disease is early blight caused by the fungus Alternaria solani, Nitrogen (N) is typically affected the most in potato tubers when they are affected by early blight; High humidity or prolonged leaf wetness can create favorable conditions for the growth and spread of early blight; if the soil pH is too acidic (below 5.0) or too alkaline (above 7.0), it can impact the overall health and vigor of the potato plant, making it more susceptible to early blight infection.','Select a late-season variety with a lower susceptibility to early blight; Time irrigation to minimize leaf wetness duration during cloudy weather and allow sufficient time for leaves to dry prior to nightfall; Avoid nitrogen and phosphorus deficiency; Scout fields regularly for infection beginning after plants reach 12 inches in height; Pay particular attention to edges of fields that are adjacent to fields planted to potato the previous year; Rotate foliar fungicide; Kill vines two to three weeks prior to harvest to allow adequate skin set; Store tubers under conditions that promote wound healing (fresh air, 95 to 99 percent relative humidity, and temperatures of 55 to 60 F) for two to three weeks after harvest; Following wound healing; store tubers in a dark, dry, and well-ventilated location gradually cooled to a temperature appropriate for the desired market; Rotate fields to non-host crops for at least three years (three to four-year crop rotation).','Potato','Commercial solutions: Shop Foliar Fungicide: https://www.amazon.in/Bayer-Folicur-Systemic-Fungicide-Litre/dp/B088GN3FS5?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&psc=1&smid=A1ZSKCLHK592D5 ; Home remedies: promptly removing and destroying the affected leaves as soon as symptoms are noticable can slow the progression of the disease; Baking soda (sodium bicarbonate) can create an alkaline environment that inhibits the growth of the fungus, Mix 1 tablespoon of baking soda with 1 gallon of water and a few drops of liquid soap Spray this mixture on your potato plants, covering both sides of the leaves. Apply every 7-10 days as a preventive measure and for early intervention; Neem oil has antifungal properties and can be used as a foliar spray to help control and slow down the spread of disease. Dilute neem oil according to the package instructions and apply it to the affected plants; Milk solution can help suppress early blight. Mix one part milk with nine parts water and spray it on the affected foliage. This remedy is considered more of a preventive measure and may not completely cure established infections; Copper-based fungicides, such as Bordeaux mixture, can be effective against early blight. Mix copper sulfate and hydrated lime with water according to the products instructions and spray it on the affected plants. Copper can reduce the severity of the disease but may not completely cure it; Garlic has natural antifungal properties. Crush several garlic cloves and steep them in water for a few days. Strain the liquid and dilute it with water. Spray this garlic solution on the potato plants. This may help slow down the progression of early blight; Applying a thick layer of organic mulch, such as straw or wood chips, around your potato plants can help prevent soil splash onto the leaves, which can carry the disease; In severe cases, especially if the disease is widespread, you may need to resort to commercial fungicides.'),
    ('LB002','Late Blight','This disease is late blight caused by the fungal pathogen Phytophthora infestans, Potassium (K) is often affected the most in potatoes when affected by late blight; High humidity or prolonged leaf wetness can create favorable conditions for the late blight pathogen to thrive and infect potato plants. Overhead irrigation, rain, or high relative humidity in the field can promote the spread and severity of late blight; Late blight tends to occur more frequently and severely in soils with a slightly acidic to neutral pH range of 5.0 to 7.0. Acidic soils (low pH) can inhibit the growth of Phytophthora infestans, while alkaline soils (high pH) can make it difficult for potato plants to take up certain nutrients, which can affect their overall resistance to diseases, including late blight.','Use potato tubers for seed from disease-free areas to ensure that the pathogen is not carried through seed tuber; The infected plant material in the field should be properly destroyed; Grow resistant varieties like Kufri Navtal; Fungicidal sprays on the appearance of initial symptoms; Spraying should be done with Dithane M-45 or Dithane Z-78 (2.5 kg/I 000 litres of water per hectare), Spraying should be repeated at 10-12 days intervals; Crop Rotation;Proper irrigation.','Potato','Commercial solutions: Shop Dithane M-45: https://shop.plantix.net/en/products/pesticides/6fbd836d-7aca-4cdd-acbe-3a3c21449fc1/dithane-m-45-dow-/?srsltid=AfmBOoq2xJHHDq3h0KvtsSkZ_mokNhEnETtdhBWRrwS6MGJAmq8jGDxFnzU, https://kisancenter.in/product/23167028/Tata-M45-Fungicide--Mancozeb-75--WP-?vid=996850 ; Shop Dithane Z-78: https://agrosiaa.com/products/detail/indofil-z-78-contact-fungicide-250gm?gclid=CjwKCAjwloynBhBbEiwAGY25dBwI0dF0Ky55MQJ3h09qZKm3U4jf_LIW8N-RmTA6A-ZPByrA-Sq9KhoCba0QAvD_BwE; Home remedies: Home remedies are generally more effective as preventive measures or for early intervention; Copper-based fungicides, such as Bordeaux mixture, can help slow the spread of late blight. Mix copper sulfate and hydrated lime with water according to the products instructions. Spray this mixture onto the affected plants and surrounding foliage. Copper can reduce the severity of the disease but may not completely cure it; If you notice late blight symptoms on your potato plants, promptly remove and destroy (do not compost) the infected leaves and stems. This can help prevent the disease from spreading to healthy parts of the plant; As a preventive measure or for early intervention, you can make a baking soda solution to create an unfavorable environment for late blight. Mix 1 tablespoon of baking soda, 1 gallon of water, and a few drops of liquid soap. Spray this solution on the potato plants, focusing on the affected areas. Repeat every 7-10 days; Neem oil has antifungal properties and can be used as a foliar spray to slow down the spread of disease. Dilute neem oil according to the package instructions and apply it to the affected plants; Milk solution can help suppress late blight. Mix one part milk with nine parts water and spray it on the affected foliage. This remedy is considered more of a preventive measure and may not cure established infections; Garlic has natural antifungal properties. Crush several garlic cloves and steep them in water for a few days. Strain the liquid and dilute it with water. Spray this garlic solution on the potato plants. This may help slow down the progression of late blight; Remember that while these home remedies may offer some relief, they are not guaranteed to completely cure late blight; If late blight has already spread extensively in your potato crop, its crucial to consider professional advice or commercial fungicides, as home remedies may not be sufficient to save your plants and prevent further damage.')
]

c.executemany('INSERT INTO diseases (disease_code,disease_name,reason,measures,crop,suggestions) VALUES (?,?,?,?,?,?)', diseases_data)
conn.commit()
conn.close()
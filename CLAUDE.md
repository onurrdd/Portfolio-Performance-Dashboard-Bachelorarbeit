# CLAUDE.md

* Projenin base halinde hiçbir değişiklik yapmayacaksın. AI Risk Analysis kısmında da asla değişiklik yapmayacaksın. Sadece bachelorarbeit ile ilgili kısımlarında  (Naive LLM ve LLM mit RAG vs.) değişiklik yapılacak. 

* LLM'e gönderilen tüm prompt'lar (Naive LLM ve LLM mit RAG sekmeleri) **her zaman İngilizce** olacak. Prompt metinlerini asla başka bir dilde yazma. LLM'in **cevap dili** ise ayrı bir `RESPONSE_LANGUAGE` değişkeniyle ("de" / "en" / "tr") seçilebilir olmalı; bu seçim prompt'un sonuna eklenen İngilizce bir talimat cümlesiyle ("Please respond in ...") yapılır. İlgili yerler: [callbacks/naive_llm.py](callbacks/naive_llm.py) ve [prompts.py](prompts.py) (RAG sekmesi bunu kullanır).

* AŞAMALAR

    1) Performans anomali günlerini ve sebep ticker'ları belirlemek (Bunu yaptık.)
    2) RAG mimarisinin kurulması
    3) RAG mimarisine önemli haber kaynaklarından anomali günlerine ait bilgi kaynaklarının çekilmesi
    4) RAGAS'ın kurulması

* Projemi Bachelorarbeit'ım için yapıyorum. Planım şu şekilde:
    1) Bir bölümü implement et. 
    2) Implement ettiğin kısmı neler yaptığını Konzeption und Implementierung bölümünde anlat.  
    3) Implement ettiğin kısımda kullandığın teknolojilerin tekniklerin vs. teorisini Theoretische Grundlagen kısmında anlat. 

    1)i zaten sana yaptırıyorum. 2) için implementierung_schritte.md dosya oluşturup orada AŞAMALAR kısmında söylediğim aşamalara göre not et. 
    3) için kullandığımız Inhaltverzeichnis olarak kullanacağım şekilde o aşama için yazmam gereken bölümleri ayrı bir theoretische_grundlagen.md dosyasında listele.

    Her önemli değişiklikte bu implementierung_schritte.md ve theoretische_grundlagen.md dosyasını güncellersin. Bazen de ben manuel güncelleyeceğim. Bu iki dosya gitignore'da olsun.


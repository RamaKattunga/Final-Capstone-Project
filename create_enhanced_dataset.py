import pandas as pd
import random
import requests
import time
from pathlib import Path

def create_enhanced_abstracts(df):
    """
    Create longer, more detailed abstracts for better summarization
    """
    enhanced_data = []
    
    # Extended abstract templates for each category
    abstract_extensions = {
        'Technology': [
            " This research contributes to the field by introducing novel algorithmic approaches that significantly improve computational efficiency. The methodology involves systematic experimentation with multiple datasets, demonstrating consistent performance gains across various metrics. Implementation details include comprehensive benchmarking against state-of-the-art methods, revealing substantial improvements in processing speed and accuracy. The findings have important implications for practical applications in industry settings, particularly in scenarios requiring real-time processing capabilities. Future work will focus on extending these techniques to handle larger-scale problems and exploring integration with emerging technologies.",
            " The study presents innovative solutions addressing current limitations in existing approaches. Experimental validation demonstrates the effectiveness of the proposed methods through extensive testing on benchmark datasets. The research methodology includes both theoretical analysis and practical implementation, ensuring robustness and reliability of the results. Performance evaluation shows significant improvements in key metrics compared to traditional approaches. The work opens new directions for research and provides practical tools for practitioners in the field.",
            " This investigation explores cutting-edge techniques that advance the current state of knowledge in the domain. The comprehensive experimental design includes multiple evaluation scenarios to ensure generalizability of findings. Results indicate substantial improvements over existing methods, with detailed analysis of computational complexity and scalability considerations. The research provides both theoretical insights and practical contributions that can be immediately applied in real-world scenarios."
        ],
        'Healthcare': [
            " This clinical study involved a comprehensive analysis of patient outcomes across multiple treatment modalities. The research methodology included randomized controlled trials with careful attention to statistical significance and clinical relevance. Data collection spanned multiple healthcare institutions to ensure diverse patient populations and enhance generalizability of findings. The study protocol was designed to minimize bias and maximize the reliability of clinical observations. Follow-up assessments were conducted over extended periods to capture long-term effects and safety profiles. The findings provide evidence-based recommendations for clinical practice and highlight important considerations for patient care protocols.",
            " The research employs rigorous epidemiological methods to investigate disease patterns and treatment effectiveness. Patient recruitment followed strict inclusion and exclusion criteria to ensure homogeneous study populations while maintaining clinical relevance. Data analysis incorporated advanced statistical techniques to account for confounding variables and ensure robust conclusions. The study design includes both prospective and retrospective components to provide comprehensive insights into disease progression and treatment outcomes. Results contribute significantly to evidence-based medicine and inform clinical decision-making processes.",
            " This biomedical investigation combines laboratory research with clinical observations to provide translational insights into disease mechanisms and therapeutic interventions. The study protocol integrates multiple research methodologies including molecular analysis, clinical assessments, and population-based studies. Findings demonstrate the interconnection between basic scientific discoveries and clinical applications, providing a bridge between bench research and bedside practice."
        ],
        'Finance': [
            " This comprehensive financial analysis incorporates advanced econometric models and empirical testing using extensive market data spanning multiple economic cycles. The research methodology includes sophisticated risk assessment techniques and portfolio optimization strategies. Data sources encompass global financial markets with particular attention to emerging market dynamics and regulatory considerations. The study examines both theoretical frameworks and practical implementation challenges in real-world trading environments. Findings provide actionable insights for investment professionals and contribute to the broader understanding of market efficiency and behavioral finance principles.",
            " The research investigates complex financial relationships using cutting-edge analytical techniques and comprehensive market data. Methodology includes extensive backtesting of trading strategies across different market conditions and time periods. The study incorporates risk management principles and regulatory compliance considerations essential for practical implementation. Results demonstrate the effectiveness of proposed approaches in generating consistent returns while managing downside risk. The work provides valuable insights for both academic researchers and industry practitioners.",
            " This quantitative finance study employs advanced mathematical models and statistical techniques to analyze market behavior and investment opportunities. The research design includes comprehensive data analysis covering multiple asset classes and geographic regions. Findings contribute to the understanding of market dynamics and provide practical tools for financial decision-making in institutional and individual investment contexts."
        ],
        'Education': [
            " This educational research study employs mixed-methods approaches combining quantitative analysis with qualitative insights from classroom observations and student feedback. The research design includes controlled experiments comparing traditional teaching methods with innovative pedagogical approaches. Data collection involves multiple stakeholders including students, teachers, and administrators across diverse educational settings. The study examines both immediate learning outcomes and long-term retention rates to provide comprehensive understanding of educational effectiveness. Findings contribute to evidence-based teaching practices and inform curriculum development decisions. The research has important implications for educational policy and professional development programs for educators.",
            " The investigation explores innovative teaching methodologies and their impact on student engagement and learning outcomes. Research methodology includes longitudinal studies tracking student progress over extended periods, with careful attention to diverse learning styles and individual differences. The study incorporates technology integration and examines its role in enhancing educational experiences. Results provide insights into effective instructional design and highlight best practices for educational institutions seeking to improve student success rates.",
            " This comprehensive educational study examines the relationship between teaching practices, student motivation, and academic achievement. The research design includes multiple assessment methods to capture various aspects of learning and development. Findings contribute to understanding of effective pedagogical strategies and provide practical recommendations for educational improvement initiatives."
        ],
        'Environment': [
            " This environmental research study employs interdisciplinary approaches combining field observations, laboratory analysis, and computational modeling to understand complex ecological systems. The research methodology includes extensive data collection from multiple geographic locations and temporal scales to capture environmental variability. The study integrates climate data, biodiversity assessments, and human impact evaluations to provide comprehensive understanding of ecosystem dynamics. Findings have important implications for conservation strategies and environmental policy development. The research contributes to sustainable development goals and provides actionable recommendations for environmental management practices. Long-term monitoring protocols established in this study will continue to inform future research and policy decisions.",
            " The investigation examines environmental challenges using advanced analytical techniques and comprehensive field studies. Research methodology includes collaboration with multiple institutions and stakeholders to ensure broad perspective and practical relevance. The study addresses both immediate environmental concerns and long-term sustainability considerations. Results provide evidence-based recommendations for environmental protection and resource management strategies.",
            " This environmental science study combines theoretical frameworks with practical field applications to address pressing ecological challenges. The research design incorporates multiple measurement techniques and analytical approaches to ensure comprehensive understanding of environmental processes. Findings contribute to both scientific knowledge and practical solutions for environmental conservation and sustainable resource utilization."
        ]
    }
    
    for _, row in df.iterrows():
        category = row['category']
        original_abstract = row['abstract']
        
        # Choose a random extension for this category
        if category in abstract_extensions:
            extension = random.choice(abstract_extensions[category])
            enhanced_abstract = original_abstract + extension
        else:
            enhanced_abstract = original_abstract
        
        # Add metadata
        enhanced_row = row.to_dict()
        enhanced_row['enhanced_abstract'] = enhanced_abstract
        enhanced_row['author'] = f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez'])}, et al."
        enhanced_row['year'] = random.randint(2018, 2024)
        enhanced_row['citations'] = random.randint(0, 150)
        enhanced_row['word_count'] = len(enhanced_abstract.split())
        
        enhanced_data.append(enhanced_row)
    
    return pd.DataFrame(enhanced_data)

def create_sample_documents():
    """
    Create sample full-text research documents for analysis
    """
    sample_documents = {
        'tech_paper_sample.txt': """
        Title: Advanced Machine Learning Techniques for Autonomous System Navigation

        Abstract: This paper presents a comprehensive study of machine learning applications in autonomous navigation systems. We propose novel deep learning architectures that significantly improve navigation accuracy in complex environments. The research demonstrates substantial improvements over existing methods through extensive experimentation and validation.

        Introduction:
        Autonomous navigation represents one of the most challenging applications of artificial intelligence and machine learning. The complexity of real-world environments, combined with the need for real-time decision making, creates unique computational and algorithmic challenges. Recent advances in deep learning and computer vision have opened new possibilities for developing more robust and efficient navigation systems.

        The primary contribution of this work is the development of a novel neural network architecture that combines convolutional neural networks (CNNs) for visual perception with recurrent neural networks (RNNs) for temporal reasoning. This hybrid approach enables the system to build comprehensive environmental understanding while maintaining computational efficiency required for real-time applications.

        Methodology:
        Our approach consists of three main components: perception, planning, and control. The perception module utilizes advanced computer vision techniques to process sensor data from multiple sources including cameras, LiDAR, and ultrasonic sensors. The planning module employs reinforcement learning algorithms to determine optimal navigation strategies. The control module implements the planned actions using robust control theory principles.

        We conducted extensive experiments in both simulated and real-world environments. The simulation environment included various challenging scenarios such as dynamic obstacles, changing weather conditions, and different terrain types. Real-world testing was performed using multiple autonomous vehicles in controlled environments.

        Results:
        Experimental results demonstrate significant improvements in navigation accuracy, obstacle avoidance, and computational efficiency. The proposed system achieved 94% navigation success rate in complex environments, compared to 78% for baseline methods. Processing time was reduced by 35% while maintaining comparable accuracy levels.

        The system showed particular strength in handling dynamic environments where obstacles change position during navigation. Traditional path planning methods often fail in such scenarios, but our approach adapts continuously to environmental changes.

        Discussion:
        The results indicate that combining different neural network architectures can lead to substantial improvements in autonomous navigation performance. The hybrid CNN-RNN approach effectively leverages the strengths of both architectures while mitigating their individual limitations.

        One important finding is that the system's performance scales well with increased environmental complexity. This suggests that the proposed approach may be suitable for deployment in real-world applications where environmental conditions are highly variable and unpredictable.

        Conclusion:
        This research demonstrates the potential of advanced machine learning techniques for autonomous navigation systems. The proposed hybrid neural network architecture provides significant improvements in accuracy and efficiency compared to existing methods. Future work will focus on extending the approach to handle more complex scenarios and investigating potential applications in other domains.

        The findings have important implications for the development of autonomous vehicles, robotics, and other applications requiring sophisticated navigation capabilities. The research contributes to the growing body of knowledge in artificial intelligence and provides practical tools for engineers and researchers working in autonomous systems.
        """,
        
        'health_study_sample.txt': """
        Title: Effectiveness of Digital Health Interventions in Chronic Disease Management: A Comprehensive Clinical Study

        Abstract: This study evaluates the effectiveness of digital health interventions in managing chronic diseases through a randomized controlled trial involving 500 patients. Results demonstrate significant improvements in patient outcomes, medication adherence, and healthcare utilization. The research provides evidence for the integration of digital health tools in clinical practice.

        Introduction:
        Chronic diseases represent a major challenge for healthcare systems worldwide, affecting millions of patients and consuming substantial healthcare resources. Traditional disease management approaches often struggle with issues such as patient adherence, continuous monitoring, and personalized care delivery. Digital health interventions offer promising solutions to these challenges by providing continuous patient engagement and real-time health monitoring capabilities.

        The proliferation of smartphones, wearable devices, and health applications has created new opportunities for delivering healthcare services outside traditional clinical settings. These technologies enable continuous patient monitoring, personalized health recommendations, and improved communication between patients and healthcare providers.

        This study aims to evaluate the effectiveness of a comprehensive digital health platform in managing chronic diseases, specifically focusing on diabetes, hypertension, and cardiovascular disease. The research investigates both clinical outcomes and patient experience measures to provide comprehensive assessment of digital intervention effectiveness.

        Methods:
        We conducted a randomized controlled trial involving 500 patients diagnosed with chronic diseases across multiple healthcare facilities. Participants were randomly assigned to either the intervention group (digital health platform) or control group (standard care). The study duration was 12 months with regular follow-up assessments.

        The digital health intervention included a mobile application for symptom tracking, medication reminders, educational content, and communication with healthcare providers. Patients also received wearable devices for continuous monitoring of vital signs and activity levels. Healthcare providers had access to a dashboard displaying patient data and automated alerts for concerning trends.

        Primary outcomes included clinical indicators such as blood pressure, blood glucose levels, and cardiovascular risk markers. Secondary outcomes included medication adherence, healthcare utilization, patient satisfaction, and quality of life measures. Data collection included clinical assessments, patient surveys, and analysis of platform usage data.

        Results:
        The intervention group showed significant improvements across multiple clinical outcomes compared to the control group. Blood pressure control improved by 23% in hypertensive patients, while diabetic patients demonstrated 18% improvement in glycemic control. Medication adherence increased by 31% in the intervention group.

        Healthcare utilization patterns showed interesting changes with the intervention group having 40% fewer emergency department visits and 25% fewer hospital readmissions. However, primary care visits increased by 15%, suggesting more proactive healthcare engagement.

        Patient satisfaction scores were significantly higher in the intervention group, with 89% of participants reporting improved understanding of their condition and 85% expressing satisfaction with the digital health platform. Platform engagement remained high throughout the study period, with average daily usage of 12 minutes.

        Discussion:
        The results demonstrate clear benefits of digital health interventions in chronic disease management. The improvements in clinical outcomes suggest that continuous monitoring and patient engagement can lead to better disease control. The reduction in emergency healthcare utilization indicates potential cost savings for healthcare systems.

        The high level of patient engagement with the digital platform suggests that such interventions are acceptable and valuable to patients. The combination of educational content, monitoring tools, and healthcare provider communication appears to create a comprehensive support system for chronic disease management.

        One important finding is that the benefits persisted throughout the study period, suggesting that digital interventions can maintain their effectiveness over time. This is particularly important for chronic disease management, which requires long-term behavior change and adherence.

        Conclusion:
        Digital health interventions show significant promise for improving chronic disease management outcomes. The study provides strong evidence for the integration of digital health tools in clinical practice. Healthcare organizations should consider implementing comprehensive digital health platforms to support chronic disease management programs.

        Future research should focus on long-term outcomes, cost-effectiveness analysis, and optimization of digital intervention components. The findings contribute to the growing evidence base supporting digital health adoption in healthcare delivery.
        """,
        
        'finance_report_sample.txt': """
        Title: Market Volatility Analysis and Risk Assessment in Post-Pandemic Financial Markets

        Abstract: This comprehensive analysis examines market volatility patterns and risk factors in global financial markets following the COVID-19 pandemic. The study employs advanced econometric models to identify key drivers of market instability and provides risk assessment frameworks for investment decision-making. Findings reveal significant structural changes in market behavior with important implications for investors and regulators.

        Executive Summary:
        Global financial markets have experienced unprecedented volatility since the onset of the COVID-19 pandemic in early 2020. Traditional risk models and volatility forecasting techniques have shown limited effectiveness in predicting market behavior during this period of extraordinary uncertainty. This analysis provides comprehensive examination of market volatility patterns and develops updated risk assessment frameworks suitable for the current market environment.

        The research analyzes data from major global financial markets including equity indices, bond markets, currency markets, and commodity markets. The study period covers January 2020 through December 2023, encompassing the initial pandemic shock, subsequent recovery phases, and ongoing market adjustments.

        Key findings indicate that market volatility has increased significantly across all asset classes, with equity markets showing the most pronounced changes. The traditional relationship between different asset classes has been disrupted, requiring updated portfolio management strategies and risk assessment approaches.

        Market Analysis:
        Equity markets experienced extreme volatility during the initial pandemic period, with the S&P 500 declining over 30% in March 2020 before recovering to new highs by the end of the year. This rapid recovery was unprecedented in historical context and was primarily driven by massive fiscal and monetary policy interventions.

        The analysis reveals that market volatility patterns have fundamentally changed compared to pre-pandemic periods. Traditional volatility clustering effects have intensified, and the persistence of volatility shocks has increased significantly. This suggests that markets are taking longer to return to normal volatility levels following major shocks.

        Bond markets have also experienced unusual behavior, with government bond yields showing extreme movements despite central bank interventions. The typical inverse relationship between equity and bond markets has been disrupted during several periods, reducing the diversification benefits of traditional portfolio strategies.

        Currency markets have shown increased sensitivity to policy announcements and economic data releases. The US dollar's role as a safe haven currency has been reinforced during periods of market stress, but emerging market currencies have experienced significant volatility due to capital flow reversals and domestic policy uncertainty.

        Risk Assessment Framework:
        Traditional risk models based on historical volatility and correlation patterns have proven inadequate for the current market environment. This analysis proposes an enhanced risk assessment framework that incorporates regime-switching models and stress testing scenarios specifically designed for pandemic-related market disruptions.

        The new framework includes several key components: dynamic volatility modeling that adjusts to changing market conditions, correlation analysis that accounts for structural breaks, and scenario analysis that incorporates extreme events similar to the pandemic shock.

        Backtesting results demonstrate that the enhanced framework provides more accurate risk assessments compared to traditional approaches. The model shows particular improvement in capturing tail risk events and providing early warning signals for potential market disruptions.

        Investment Implications:
        The analysis provides several important implications for investment strategy and portfolio management. Traditional diversification strategies need to be updated to account for increased correlation during stress periods. Alternative investments and defensive strategies should play larger roles in portfolio construction.

        Risk budgeting approaches need to incorporate the possibility of extreme events similar to the pandemic shock. This requires maintaining higher cash reserves and implementing more sophisticated hedging strategies. Dynamic asset allocation strategies that can quickly adjust to changing market conditions are recommended over static approaches.

        The research also highlights the importance of behavioral factors in driving market volatility. Investor sentiment and social media influence have become increasingly important in determining short-term market movements. Investment strategies should incorporate these factors in addition to traditional fundamental and technical analysis.

        Regulatory Considerations:
        The analysis reveals several areas where regulatory attention may be warranted. Market structure changes during periods of extreme volatility raise questions about market stability and investor protection. The role of high-frequency trading and algorithmic trading systems in amplifying volatility deserves further investigation.

        Central bank policy effectiveness has been demonstrated during the pandemic, but the long-term consequences of unprecedented monetary policy interventions remain uncertain. Regulatory frameworks should be updated to address potential systemic risks arising from current policy approaches.

        Conclusion:
        Financial markets have undergone fundamental changes following the COVID-19 pandemic, requiring updated approaches to risk assessment and investment strategy. Traditional models and frameworks need to be enhanced to account for new sources of volatility and correlation patterns.

        The research provides practical tools and frameworks for navigating the current market environment. Investors and financial institutions should update their risk management practices to incorporate the findings and recommendations presented in this analysis.

        Future research should continue monitoring market developments and refining risk models as more data becomes available. The financial industry must remain adaptive and responsive to ongoing changes in market structure and behavior patterns.
        """,
        
        'education_paper_sample.txt': """
        Title: Impact of Technology Integration on Student Learning Outcomes in Higher Education: A Longitudinal Study

        Abstract: This longitudinal study examines the impact of technology integration on student learning outcomes across multiple academic disciplines in higher education. The research follows 1,200 students over four years to assess the effectiveness of various educational technologies. Results demonstrate significant improvements in student engagement, academic performance, and digital literacy skills.

        Introduction:
        The integration of technology in higher education has accelerated rapidly over the past decade, with the COVID-19 pandemic serving as a catalyst for widespread adoption of digital learning tools. Universities worldwide have invested heavily in learning management systems, online collaboration platforms, and interactive educational technologies. However, empirical evidence regarding the effectiveness of these investments in improving student learning outcomes has been limited and inconsistent.

        This study addresses the gap in longitudinal research on technology integration effectiveness by following a large cohort of students throughout their undergraduate experience. The research examines multiple dimensions of learning outcomes including academic performance, student engagement, critical thinking skills, and digital literacy development.

        The study is particularly important as higher education institutions continue to make significant investments in educational technology while facing pressure to demonstrate return on investment and improve student success rates. Understanding which technologies provide the greatest benefit and under what conditions they are most effective is crucial for informed decision-making in educational technology adoption.

        Literature Review:
        Previous research on educational technology effectiveness has produced mixed results, with some studies showing significant positive effects while others find no significant impact on learning outcomes. Meta-analyses suggest that the effectiveness of educational technology depends heavily on implementation quality, instructor training, and integration with pedagogical approaches.

        Studies have identified several factors that influence technology effectiveness including student digital literacy levels, institutional support for technology adoption, and alignment between technology features and learning objectives. The importance of instructor attitudes and competencies in technology use has been consistently highlighted across multiple studies.

        Research has also shown that different types of educational technologies may be more effective for different learning objectives and student populations. Interactive simulation software has shown particular promise for STEM education, while collaborative platforms have demonstrated effectiveness in developing communication and teamwork skills.

        Methodology:
        This longitudinal study followed 1,200 undergraduate students from their first year through graduation across three major universities. Students were enrolled in various academic disciplines including STEM fields, humanities, business, and social sciences. The study employed a mixed-methods approach combining quantitative analysis of academic performance data with qualitative assessment of student experiences.

        Technology interventions included learning management systems with advanced analytics, interactive simulation software, collaborative project platforms, virtual reality applications for immersive learning, and adaptive learning systems that personalize content delivery based on individual student progress.

        Data collection included academic performance metrics (GPA, course completion rates, graduation rates), standardized assessments of critical thinking and problem-solving skills, digital literacy evaluations, and student satisfaction surveys. Faculty interviews and classroom observations provided additional qualitative insights into implementation effectiveness.

        The study design included both treatment and control groups to enable comparison between traditional and technology-enhanced learning environments. Random assignment was used where possible, though some analyses relied on quasi-experimental designs due to practical constraints.

        Results:
        Academic performance improvements were observed across all technology intervention groups compared to control groups. Average GPA increases ranged from 0.2 to 0.4 points depending on the specific technology and implementation approach. Course completion rates improved by 8-15% in technology-enhanced courses.

        Student engagement metrics showed substantial improvements with technology integration. Time spent on learning activities increased by an average of 25% in technology-enhanced courses. Student participation in class discussions and collaborative activities improved significantly when supported by appropriate technology platforms.

        Critical thinking and problem-solving skills showed modest but statistically significant improvements in students exposed to interactive simulation and problem-based learning technologies. These improvements were most pronounced in STEM disciplines where hands-on experimentation and visualization tools were extensively used.

        Digital literacy skills improved dramatically across all student groups exposed to educational technology interventions. Students demonstrated increased proficiency in information evaluation, digital communication, and technology problem-solving. These skills transferred effectively to other courses and professional contexts.

        Long-term outcomes showed that students who experienced extensive technology integration had higher graduation rates (89% vs. 82% for control groups) and reported greater satisfaction with their educational experience. Follow-up surveys of graduates indicated that technology-enhanced education better prepared them for workplace demands.

        Discussion:
        The results provide strong evidence that thoughtful integration of educational technology can significantly improve student learning outcomes in higher education. The key finding is that technology effectiveness depends heavily on implementation quality and integration with sound pedagogical principles rather than the technology itself.

        Successful implementations were characterized by comprehensive instructor training, institutional support for technology adoption, and careful alignment between technology features and learning objectives. Institutions that treated technology as a supplement to rather than replacement for effective teaching practices achieved the best results.

        The study also revealed important differences in technology effectiveness across academic disciplines and student populations. STEM fields showed the greatest benefits from simulation and visualization technologies, while humanities and social science courses benefited more from collaborative platforms and multimedia content creation tools.

        Student digital literacy development emerged as an important secondary benefit of educational technology integration. These skills have become increasingly important for academic success and career preparation, suggesting that technology integration provides value beyond traditional learning outcomes.

        Implications for Practice:
        The findings provide several important implications for higher education institutions considering educational technology investments. Successful technology integration requires comprehensive planning that includes instructor training, technical support, and ongoing evaluation of effectiveness.

        Institutions should adopt a strategic approach to technology selection that considers specific learning objectives, student populations, and existing infrastructure. Rather than pursuing technology for its own sake, institutions should focus on technologies that demonstrably improve specific learning outcomes.

        Professional development for faculty emerges as a critical success factor. Institutions should invest in comprehensive training programs that help faculty understand how to effectively integrate technology into their teaching practices. Ongoing support and mentoring are essential for sustained implementation success.

        Conclusion:
        This longitudinal study provides compelling evidence that educational technology can significantly improve student learning outcomes when implemented thoughtfully and strategically. The research contributes to the growing body of evidence supporting technology integration in higher education while highlighting important factors that influence implementation success.

        Higher education institutions should view educational technology as a powerful tool for improving student success, but must commit to comprehensive implementation approaches that prioritize pedagogical effectiveness over technological sophistication. The findings support continued investment in educational technology while emphasizing the importance of focusing on proven approaches and comprehensive implementation strategies.

        Future research should continue to examine long-term outcomes of technology integration and investigate emerging technologies such as artificial intelligence and virtual reality that may offer new opportunities for enhancing student learning experiences.
        """,
        
        'environment_study_sample.txt': """
        Title: Climate Change Impacts on Biodiversity: A Comprehensive Assessment of Ecosystem Vulnerability and Adaptation Strategies

        Abstract: This comprehensive environmental study assesses the impacts of climate change on biodiversity across multiple ecosystems and geographic regions. The research combines field observations, satellite data analysis, and predictive modeling to evaluate ecosystem vulnerability and identify effective adaptation strategies. Results indicate significant threats to biodiversity with urgent need for comprehensive conservation and adaptation measures.

        Introduction:
        Climate change represents one of the most significant threats to global biodiversity in the 21st century. Rising temperatures, changing precipitation patterns, and increased frequency of extreme weather events are disrupting ecosystems worldwide and threatening the survival of countless species. Understanding these impacts and developing effective adaptation strategies is crucial for biodiversity conservation and ecosystem sustainability.

        This study provides comprehensive assessment of climate change impacts on biodiversity across terrestrial, marine, and freshwater ecosystems. The research examines current trends, projects future scenarios, and evaluates the effectiveness of various adaptation and mitigation strategies. The findings have important implications for conservation policy and environmental management practices.

        The urgency of this research is underscored by recent reports indicating that species extinction rates are accelerating and ecosystem degradation is occurring at unprecedented scales. The interconnected nature of ecological systems means that impacts on one species or ecosystem component can cascade through entire food webs, potentially leading to ecosystem collapse.

        Background and Context:
        Global temperatures have increased by approximately 1.1°C since pre-industrial times, with particularly rapid warming observed in polar and high-altitude regions. This warming has been accompanied by changes in precipitation patterns, sea level rise, ocean acidification, and shifts in seasonal timing of biological events.

        Biodiversity loss has accelerated dramatically over the past century, with current extinction rates estimated to be 100-1,000 times higher than natural background rates. Climate change is now recognized as one of the primary drivers of biodiversity loss, along with habitat destruction, pollution, and invasive species.

        Ecosystems vary significantly in their vulnerability to climate change impacts. Arctic ecosystems, coral reefs, and mountain ecosystems are particularly vulnerable due to their sensitivity to temperature changes and limited adaptation options. Forest ecosystems face threats from changing precipitation patterns, increased wildfire frequency, and pest outbreaks.

        Methodology:
        This study employed a multi-faceted approach combining field research, remote sensing data analysis, and predictive modeling. Field studies were conducted across 15 different ecosystem types in six geographic regions over a five-year period. Research sites were selected to represent major ecosystem types and climate gradients.

        Satellite data analysis covered a 30-year period and included vegetation indices, land cover changes, and surface temperature measurements. This long-term perspective enabled identification of trends and patterns that might not be apparent from shorter-term studies.

        Predictive modeling incorporated multiple climate scenarios based on different greenhouse gas emission pathways. Species distribution models were used to project future suitable habitat for key indicator species. Ecosystem models simulated changes in ecosystem structure and function under different climate scenarios.

        The research also included extensive literature review and meta-analysis of existing studies to provide broader context and validation for field observations. Collaboration with research institutions worldwide enabled data sharing and comparative analysis across different geographic regions.

        Results and Findings:
        Temperature-sensitive species showed significant population declines and range shifts toward higher latitudes and elevations. Arctic species, including polar bears, arctic foxes, and various seabird species, experienced substantial habitat loss due to sea ice reduction. Mountain species faced habitat compression as suitable climate zones shifted upward.

        Forest ecosystems demonstrated varying responses to climate change depending on species composition and geographic location. Boreal forests showed increased tree mortality due to pest outbreaks and drought stress. Tropical forests experienced changes in species composition and reduced carbon storage capacity in some regions.

        Marine ecosystems showed widespread impacts including coral bleaching events, changes in fish distribution patterns, and disruption of marine food chains. Ocean acidification caused additional stress on shell-forming organisms and coral reefs. Coastal ecosystems faced increased pressure from sea level rise and storm surge intensity.

        Freshwater ecosystems experienced significant alterations in hydrology, water temperature, and seasonal patterns. Many rivers and lakes showed changes in seasonal flow patterns, affecting aquatic species reproduction and migration cycles. Wetland ecosystems faced increased pressure from altered precipitation patterns and human water use demands.

        Species with limited dispersal abilities or highly specialized habitat requirements showed the greatest vulnerability to climate change impacts. Endemic species in isolated habitats, such as mountaintop or island ecosystems, faced particularly high extinction risks due to limited adaptation options.

        Adaptation Strategies:
        The research identified several effective adaptation strategies for biodiversity conservation under climate change. Habitat corridor establishment emerged as a critical strategy for enabling species movement and genetic exchange. Protected area networks need expansion and connectivity enhancement to facilitate species range shifts.

        Ecosystem restoration projects showed promise for enhancing ecosystem resilience and adaptation capacity. Restoration of degraded habitats can provide stepping stones for species movement and increase overall ecosystem stability. Native species reintroduction programs have shown success in some contexts but require careful planning and monitoring.

        Assisted migration, or the human-facilitated movement of species to more suitable habitats, represents a controversial but potentially necessary adaptation strategy for some highly threatened species. This approach requires careful consideration of ecological risks and extensive research to avoid unintended consequences.

        Ex-situ conservation measures, including seed banks, captive breeding programs, and genetic resource preservation, provide important backup options for species facing immediate extinction threats. These approaches are particularly important for species with small population sizes or highly restricted habitat requirements.

        Policy and Management Implications:
        The findings highlight the urgent need for comprehensive policy responses that address both climate change mitigation and biodiversity conservation. Current conservation strategies need updating to account for dynamic climate conditions and shifting species distributions.

        International cooperation is essential for effective biodiversity conservation under climate change, as species migrations and ecosystem changes cross political boundaries. Existing international agreements and frameworks need strengthening and better coordination to address transboundary conservation challenges.

        Funding for conservation research and implementation needs substantial increase to match the scale of the challenge. Current conservation budgets are inadequate for implementing necessary adaptation measures across the global scale required for effective biodiversity protection.

        Adaptive management approaches that can respond to changing conditions and new information are essential for conservation success under climate uncertainty. Traditional static conservation approaches are insufficient for dynamic climate conditions and require fundamental revision.

        Future Research Directions:
        Continued monitoring and research are essential for understanding evolving climate change impacts and evaluating adaptation strategy effectiveness.         Long-term ecological monitoring programs need expansion and standardization to provide consistent data for global analysis.

        Emerging technologies including environmental DNA sampling, automated sensor networks, and artificial intelligence for species identification offer new opportunities for biodiversity monitoring and research. These technologies can significantly enhance data collection capacity and analytical capabilities.

        Research on ecosystem tipping points and cascade effects requires priority attention to identify critical thresholds and early warning systems. Understanding non-linear ecosystem responses is essential for predicting and preventing ecosystem collapse.

        Conclusion:
        Climate change poses unprecedented threats to global biodiversity requiring immediate and comprehensive response. The research demonstrates that significant impacts are already occurring across all major ecosystem types, with more severe impacts projected under continued warming scenarios.

        Effective biodiversity conservation under climate change requires integrated approaches that combine traditional conservation strategies with climate adaptation measures. Success depends on substantial increases in conservation funding, international cooperation, and political commitment to addressing both climate change and biodiversity loss.

        The window of opportunity for preventing catastrophic biodiversity loss is rapidly closing, making immediate action essential. The research provides roadmap for effective conservation strategies, but implementation requires unprecedented global cooperation and commitment to environmental protection.

        Future generations depend on current decisions and actions to preserve Earth's biological heritage. The scientific evidence clearly demonstrates both the magnitude of the challenge and the potential for effective responses if implemented with sufficient scale and urgency.
        """
    }
    
    # Create sample_documents directory
    sample_dir = Path('data/sample_documents')
    sample_dir.mkdir(exist_ok=True)
    
    # Write sample documents
    for filename, content in sample_documents.items():
        file_path = sample_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created: {file_path}")
    
    return sample_documents

def create_enhanced_dataset():
    """
    Create enhanced research papers dataset with longer abstracts
    """
    # Load the cleaned dataset from Component 2
    try:
        df = pd.read_csv('cleaned_dataset.csv')
        print("Loaded cleaned dataset from Component 2")
    except FileNotFoundError:
        # Fallback to original dataset
        try:
            df = pd.read_csv('research_papers_dataset.csv')
            print("Loaded original dataset from Component 1")
        except FileNotFoundError:
            print("Error: No dataset found. Please run Components 1 or 2 first.")
            return None
    
    # Create enhanced version
    enhanced_df = create_enhanced_abstracts(df)
    
    # Save enhanced dataset
    enhanced_df.to_csv('enhanced_research_papers.csv', index=False)
    print(f"Enhanced dataset created with {len(enhanced_df)} papers")
    print(f"Average abstract length: {enhanced_df['word_count'].mean():.0f} words")
    
    return enhanced_df

# Create both enhanced dataset and sample documents
if __name__ == "__main__":
    print("Creating enhanced dataset and sample documents...")
    
    # Create enhanced dataset
    enhanced_data = create_enhanced_dataset()
    
    # Create sample documents
    sample_docs = create_sample_documents()
    
    print("\nData creation completed!")
    print("Files created:")
    print("- enhanced_research_papers.csv")
    print("- data/sample_documents/ (5 full-text documents)")

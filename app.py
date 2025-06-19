from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
import requests
import praw
from openai import OpenAI
from typing import List, Dict, Any
import time
import re
from urllib.parse import urlparse
import uuid
import threading
from queue import Queue
from dotenv import load_dotenv
import random
import logging
import json
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_interactions.log'),
        logging.StreamHandler()  # This will show logs in console/Render logs
    ]
)

# Create a separate logger for user interactions
interaction_logger = logging.getLogger('user_interactions')
interaction_logger.setLevel(logging.INFO)

class InteractionLogger:
    def __init__(self):
        self.logger = interaction_logger
    
    def log_interaction(self, session_id, event_type, data):
        """Log user interaction events with structured data"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session_id,
            'event_type': event_type,
            'data': data
        }
        
        # Log as JSON for easy parsing
        self.logger.info(f"USER_INTERACTION: {json.dumps(log_entry, ensure_ascii=False)}")
    
    def log_user_query(self, session_id, query):
        """Log initial user query"""
        self.log_interaction(session_id, 'USER_QUERY', {
            'query': query,
            'query_length': len(query)
        })
    
    def log_clarifying_questions(self, session_id, questions):
        """Log generated clarifying questions"""
        self.log_interaction(session_id, 'CLARIFYING_QUESTIONS_GENERATED', {
            'questions': questions,
            'question_count': len(questions)
        })
    
    def log_user_answer(self, session_id, question_index, question, answer):
        """Log user's answer to clarifying question"""
        self.log_interaction(session_id, 'USER_ANSWER', {
            'question_index': question_index,
            'question': question,
            'answer': answer,
            'answer_length': len(answer)
        })
    
    def log_enhanced_context(self, session_id, enhanced_context):
        """Log the enhanced context creation"""
        self.log_interaction(session_id, 'ENHANCED_CONTEXT_CREATED', {
            'enhanced_context': enhanced_context,
            'context_length': len(enhanced_context)
        })
    
    def log_preferences_extracted(self, session_id, preferences):
        """Log extracted user preferences"""
        self.log_interaction(session_id, 'PREFERENCES_EXTRACTED', {
            'preferences': preferences,
            'total_interests': len(preferences.get('interests_hobbies', [])),
            'has_budget': len(preferences.get('budget_preferences', [])) > 0
        })
    
    def log_optimized_queries(self, session_id, optimized_queries):
        """Log optimized search queries"""
        self.log_interaction(session_id, 'SEARCH_QUERIES_OPTIMIZED', {
            'optimized_queries': optimized_queries,
            'query_count': len(optimized_queries)
        })
    
    def log_reddit_search_results(self, session_id, reddit_posts_found, reddit_content_extracted):
        """Log Reddit search and extraction results"""
        self.log_interaction(session_id, 'REDDIT_SEARCH_COMPLETED', {
            'posts_found': len(reddit_posts_found),
            'posts_extracted': len(reddit_content_extracted),
            'reddit_posts': [{'title': post['title'], 'url': post['url'], 'relevance_score': post.get('relevance_score', 0)} for post in reddit_posts_found]
        })
    
    def log_product_recommendations(self, session_id, recommendations):
        """Log generated product recommendations"""
        self.log_interaction(session_id, 'PRODUCT_RECOMMENDATIONS_GENERATED', {
            'recommendations': recommendations,
            'recommendation_count': len(recommendations)
        })
    
    def log_recommendation_balance_check(self, session_id, balance_info):
        """Log recommendation balance verification"""
        self.log_interaction(session_id, 'RECOMMENDATION_BALANCE_VERIFIED', {
            'balance_check': balance_info
        })
    
    def log_user_selection(self, session_id, selected_indices, selected_products):
        """Log user's product selection"""
        self.log_interaction(session_id, 'USER_PRODUCT_SELECTION', {
            'selected_indices': selected_indices,
            'selected_products': selected_products,
            'selection_count': len(selected_products)
        })
    
    def log_budget_extraction(self, session_id, budget_info):
        """Log budget extraction results"""
        self.log_interaction(session_id, 'BUDGET_EXTRACTED', {
            'budget_info': budget_info,
            'has_budget': budget_info.get('has_budget', False),
            'max_amount': budget_info.get('max_amount', 0),
            'budget_display': budget_info.get('display', 'Unknown')
        })
    
    def log_final_products(self, session_id, product_results):
        """Log final product search results"""
        # Simplify product results for logging
        simplified_results = []
        total_products = 0
        budget_compliant_products = 0
        
        for category in product_results:
            category_products = []
            for product in category.get('products', []):
                product_info = {
                    'title': product.get('title', ''),
                    'url': product.get('url', ''),
                    'site': product.get('site', ''),
                    'within_budget': product.get('budget_check', {}).get('within_budget', False),
                    'extracted_prices': product.get('budget_check', {}).get('extracted_prices', [])
                }
                category_products.append(product_info)
                total_products += 1
                if product_info['within_budget']:
                    budget_compliant_products += 1
            
            simplified_results.append({
                'category': category.get('category', ''),
                'original_recommendation': category.get('original_recommendation', ''),
                'products_found': len(category.get('products', [])),
                'products': category_products,
                'budget_info': category.get('budget_info', {})
            })
        
        self.log_interaction(session_id, 'FINAL_PRODUCTS_FOUND', {
            'product_results': simplified_results,
            'total_categories': len(product_results),
            'total_products': total_products,
            'budget_compliant_products': budget_compliant_products,
            'budget_compliance_rate': budget_compliant_products / total_products if total_products > 0 else 0
        })
    
    def log_error(self, session_id, error_type, error_message, step):
        """Log errors during the process"""
        self.log_interaction(session_id, 'ERROR', {
            'error_type': error_type,
            'error_message': str(error_message),
            'step': step
        })

# Initialize the interaction logger
interaction_logger_instance = InteractionLogger()

# Load environment variables from .env file (for local development)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-fallback-secret-key-change-this')
CORS(app)

class GiftRecommendationSystem:
    def __init__(self):
        # Load API keys from environment variables (SECURE)
        self.OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
        self.GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        self.SEARCH_ENGINE_ID = os.environ.get('SEARCH_ENGINE_ID')
        
        # Reddit API Configuration from environment
        self.REDDIT_CLIENT_ID = os.environ.get('REDDIT_CLIENT_ID')
        self.REDDIT_CLIENT_SECRET = os.environ.get('REDDIT_CLIENT_SECRET')
        self.REDDIT_USER_AGENT = os.environ.get('REDDIT_USER_AGENT', 'GiftRecommendation:v1.0')

        # Validate that all required environment variables are present
        required_vars = [
            'OPENAI_API_KEY', 'GOOGLE_API_KEY', 'SEARCH_ENGINE_ID',
            'REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET'
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Initialize APIs
        try:
            self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)
            self.reddit = praw.Reddit(
                client_id=self.REDDIT_CLIENT_ID,
                client_secret=self.REDDIT_CLIENT_SECRET,
                user_agent=self.REDDIT_USER_AGENT
            )
            print("✅ APIs initialized successfully")
        except Exception as e:
            print(f"❌ API initialization error: {e}")
            raise

    def extract_preferences_and_interests(self, enhanced_context: str) -> Dict[str, List[str]]:
        """Extract and categorize user preferences to ensure balanced representation"""
        print("\n=== EXTRACTING USER PREFERENCES FOR BALANCED SEARCH ===")
        
        extraction_prompt = f"""
        Analyze the following user requirements and extract ALL mentioned preferences, interests, and characteristics into organized categories:

        USER REQUIREMENTS:
        {enhanced_context}

        Extract and categorize ALL mentioned items into these categories:

        1. INTERESTS/HOBBIES: Any activities, hobbies, or interests mentioned
        2. PERSONALITY_TRAITS: Any personality characteristics or behavioral preferences
        3. DEMOGRAPHICS: Age, gender, relationship, profession, etc.
        4. BUDGET_PREFERENCES: Any budget or price-related mentions
        5. OCCASION_CONTEXT: Birthday, anniversary, holiday, etc.
        6. STYLE_PREFERENCES: Design preferences, colors, styles mentioned
        7. FUNCTIONAL_NEEDS: Practical needs or use cases mentioned

        Return your response in this EXACT JSON format:
        {{
            "interests_hobbies": ["item1", "item2", "item3"],
            "personality_traits": ["trait1", "trait2"],
            "demographics": ["demo1", "demo2"],
            "budget_preferences": ["budget1"],
            "occasion_context": ["occasion1"],
            "style_preferences": ["style1", "style2"],
            "functional_needs": ["need1", "need2"]
        }}

        IMPORTANT: 
        - Include ALL mentioned items, even if similar
        - Use the person's exact words when possible
        - If a category has no items, use an empty list []
        - Be comprehensive and don't miss any details
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent extraction
            )

            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            preferences = json.loads(response_text)
            
            print("✅ EXTRACTED PREFERENCES:")
            for category, items in preferences.items():
                if items:
                    print(f"  {category.upper()}: {items}")
            
            return preferences
            
        except Exception as e:
            print(f"✗ Error extracting preferences: {e}")
            # Fallback: manual extraction with basic keywords
            return self._fallback_preference_extraction(enhanced_context)

    def _fallback_preference_extraction(self, enhanced_context: str) -> Dict[str, List[str]]:
        """Fallback preference extraction if AI parsing fails"""
        print("  ⚠️ Using fallback preference extraction")
        
        # Simple keyword-based extraction
        text_lower = enhanced_context.lower()
        
        interests_keywords = ['interest', 'hobby', 'like', 'love', 'enjoy', 'passion']
        extracted_interests = []
        
        # Look for patterns like "interests in X" or "likes Y"
        for keyword in interests_keywords:
            if keyword in text_lower:
                # Extract surrounding context
                parts = text_lower.split(keyword)
                if len(parts) > 1:
                    after_keyword = parts[1].split('.')[0].split(',')[0].strip()
                    if after_keyword:
                        extracted_interests.append(after_keyword)
        
        return {
            "interests_hobbies": extracted_interests[:5],  # Limit to prevent noise
            "personality_traits": [],
            "demographics": [],
            "budget_preferences": [],
            "occasion_context": [],
            "style_preferences": [],
            "functional_needs": []
        }

    def optimize_search_query(self, enhanced_context: str) -> List[str]:
        """IMPROVED: Generate balanced, diverse search queries for all preferences"""
        print("\n=== GENERATING BALANCED SEARCH QUERIES ===")

        # First, extract all user preferences
        preferences = self.extract_preferences_and_interests(enhanced_context)
        
        # Generate queries that ensure all preferences get equal representation
        optimization_prompt = f"""
        Generate 8-10 diverse, balanced search queries for Reddit gift recommendations based on these user preferences:

        USER PREFERENCES (extracted):
        {json.dumps(preferences, indent=2)}

        FULL CONTEXT:
        {enhanced_context}

        REQUIREMENTS for query generation:
        1. Create queries that cover ALL mentioned interests/hobbies equally
        2. If multiple interests mentioned, create separate queries for each major interest
        3. Ensure no single preference dominates all queries
        4. Include combination queries that merge different preferences
        5. Always include "reddit" and "India" in each query
        6. Keep queries 3-8 words maximum
        7. Focus on gift-related terms

        QUERY STRATEGY:
        - Generate 2-3 queries for EACH major interest/hobby mentioned
        - Generate 1-2 general queries combining multiple interests
        - Generate 1-2 demographic/occasion specific queries
        - Randomize the order to prevent bias

        GOOD Examples:
        - "reddit gift ideas photography enthusiast India"
        - "reddit documentary lover birthday gifts India"
        - "reddit sports fan gift recommendations India"
        - "reddit photography documentaries combo gifts India"
        - "reddit birthday brother multiple interests India"

        Generate exactly 8-10 queries, ensuring balanced coverage of ALL preferences.
        Return only the queries, one per line, no numbering or extra text.
        """

        try:
            print("🤖 Generating balanced search queries...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": optimization_prompt}],
                max_tokens=200,
                temperature=0.5  # Moderate temperature for some variation
            )

            queries_text = response.choices[0].message.content.strip()

            # Parse queries into a list
            queries = []
            for line in queries_text.split('\n'):
                line = line.strip()
                if line and 'reddit' in line.lower():
                    # Clean up any formatting
                    query = line.replace('"', '').replace('- ', '').strip()
                    queries.append(query)

            # Shuffle queries to prevent order bias in search results
            random.shuffle(queries)

            print(f"✅ GENERATED {len(queries)} BALANCED SEARCH QUERIES:")
            for i, query in enumerate(queries, 1):
                print(f"   {i}. {query}")

            return queries

        except Exception as e:
            print(f"✗ Error generating balanced queries: {e}")
            # Enhanced fallback with preference awareness
            return self._generate_fallback_balanced_queries(preferences)

    def _generate_fallback_balanced_queries(self, preferences: Dict[str, List[str]]) -> List[str]:
        """Generate fallback queries ensuring all preferences are covered"""
        print("  ⚠️ Using enhanced fallback query generation")
        
        queries = []
        
        # Generate queries for each interest/hobby
        for interest in preferences.get('interests_hobbies', []):
            queries.append(f"reddit gift ideas {interest} India")
            queries.append(f"reddit {interest} birthday gifts India")
        
        # Add general combination queries
        all_interests = preferences.get('interests_hobbies', [])
        if len(all_interests) > 1:
            # Combine interests in different ways
            queries.append(f"reddit gift ideas {' '.join(all_interests[:2])} India")
            if len(all_interests) > 2:
                queries.append(f"reddit gifts {all_interests[0]} {all_interests[-1]} India")
        
        # Add demographic/occasion queries
        demographics = preferences.get('demographics', [])
        occasions = preferences.get('occasion_context', [])
        
        if demographics:
            queries.append(f"reddit gift recommendations {demographics[0]} India")
        if occasions:
            queries.append(f"reddit {occasions[0]} gift ideas India")
        
        # Ensure minimum queries
        if len(queries) < 4:
            queries.extend([
                "reddit birthday gift ideas India",
                "reddit gift recommendations brother India",
                "reddit unique gifts India"
            ])
        
        # Shuffle to prevent bias
        random.shuffle(queries)
        
        return queries[:8]  # Limit to 8 queries

    def analyze_for_product_recommendations(self, reddit_content: List[Dict[str, Any]], enhanced_context: str) -> List[str]:
        """IMPROVED: Ensure balanced analysis of all user preferences"""
        print("\n=== ANALYZING CONTENT WITH BALANCED PREFERENCE WEIGHTING ===")

        if not reddit_content:
            print("❌ No Reddit content to analyze")
            return []

        # Extract preferences for balanced analysis
        preferences = self.extract_preferences_and_interests(enhanced_context)

        # Prepare content for analysis with URL mapping
        analysis_text = f"USER REQUIREMENTS:\n{enhanced_context}\n\n"
        analysis_text += f"EXTRACTED USER PREFERENCES (for balanced analysis):\n{json.dumps(preferences, indent=2)}\n\n"
        analysis_text += "REDDIT DISCUSSIONS CONTENT:\n"
        analysis_text += "=" * 50 + "\n"

        # Create a mapping of posts with their URLs for reference
        post_urls = {}

        for i, post in enumerate(reddit_content, 1):
            post_identifier = f"POST_{i}"
            post_urls[post_identifier] = post['url']

            analysis_text += f"\n{post_identifier}: {post['title']}\n"
            analysis_text += f"URL: {post['url']}\n"
            analysis_text += f"Subreddit: r/{post['subreddit']}\n"
            analysis_text += f"Score: {post['score']} | Comments: {post['num_comments']}\n"

            if post['selftext']:
                analysis_text += f"Post Content: {post['selftext'][:500]}...\n"

            analysis_text += f"\nTOP COMMENTS:\n"

            for j, comment in enumerate(post['comments'][:25], 1):  # Limit to top 25 comments per post for analysis
                analysis_text += f"Comment {j} (Score: {comment['score']}): {comment['body'][:250]}...\n"

            analysis_text += "\n" + "-" * 50 + "\n"

        print(f"✓ Prepared content for balanced analysis ({len(analysis_text)} characters)")

        # IMPROVED analysis prompt with explicit balance requirements
        analysis_prompt = f"""
        Analyze the Reddit discussions and provide 10 BALANCED product recommendations that represent ALL user preferences equally.

        {analysis_text}

        CRITICAL BALANCE REQUIREMENTS:
        1. **EQUAL REPRESENTATION**: Ensure each mentioned interest/hobby gets fair representation in recommendations
        2. **NO FIRST-PREFERENCE BIAS**: Don't favor the first-mentioned interest over others
        3. **DIVERSE CATEGORIES**: Include products from different categories and price ranges
        4. **PREFERENCE DISTRIBUTION**: If user mentioned 3 interests, aim for 3-4 products per interest
        5. **COMBINATION PRODUCTS**: Include items that serve multiple interests when possible

        USER'S EXTRACTED PREFERENCES:
        {json.dumps(preferences, indent=2)}

        ANALYSIS STRATEGY:
        - Review ALL Reddit discussions for mentions of each preference area
        - Give equal weight to each interest/hobby mentioned
        - Look for products that serve multiple preferences
        - Ensure variety in product types and price ranges
        - Include both specific items and broader categories

        Provide exactly 10 recommendations following this distribution strategy:
        - Distribute recommendations evenly across all mentioned interests
        - Include combination products that serve multiple interests
        - Vary price ranges and product types
        - Prioritize highly-discussed items from the Reddit threads

        Format your response as:
        1. [Product Name/Category] - [Brief explanation including how it relates to user's specific interests and preferences, with price range if mentioned] | Source: [POST_X]
        2. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        [Continue for all 10 recommendations]

        IMPORTANT: Explicitly mention which user interest(s) each recommendation addresses to ensure balanced coverage.
        """

        try:
            print("🤖 Performing balanced content analysis...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1500,
                temperature=0.4  # Moderate temperature for balanced creativity
            )

            recommendations_text = response.choices[0].message.content.strip()

            # Parse recommendations and add URLs
            recommendations = []
            for line in recommendations_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    # Extract POST_X reference and replace with actual URL
                    enhanced_line = line
                    for post_id, url in post_urls.items():
                        if post_id in line:
                            enhanced_line = line.replace(f"Source: {post_id}", f"Source: {url}")
                            break
                    recommendations.append(enhanced_line)

            # Verify balanced distribution
            self._verify_recommendation_balance(recommendations, preferences)

            print("✅ BALANCED ANALYSIS COMPLETED")
            print(f"Generated {len(recommendations)} balanced product recommendations")

            return recommendations

        except Exception as e:
            print(f"✗ Error analyzing content: {e}")
            return []

    def _verify_recommendation_balance(self, recommendations: List[str], preferences: Dict[str, List[str]]) -> None:
        """Verify that recommendations are balanced across all preferences"""
        print("\n=== VERIFYING RECOMMENDATION BALANCE ===")
        
        interests = preferences.get('interests_hobbies', [])
        if not interests:
            print("  ⚠️ No specific interests found for balance verification")
            return
        
        # Count mentions of each interest in recommendations
        interest_counts = {interest: 0 for interest in interests}
        
        for rec in recommendations:
            rec_lower = rec.lower()
            for interest in interests:
                if interest.lower() in rec_lower:
                    interest_counts[interest] += 1
        
        print("  📊 RECOMMENDATION DISTRIBUTION:")
        for interest, count in interest_counts.items():
            print(f"    {interest}: {count} recommendations")
        
        # Check for significant imbalance
        if interests and len(interests) > 1:
            counts = list(interest_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            
            if max_count > 0 and min_count == 0:
                print(f"  ⚠️ WARNING: Interest '{interests[counts.index(max_count)]}' dominates, some interests have no representation")
            elif max_count > min_count * 2:
                print(f"  ⚠️ WARNING: Significant imbalance detected (max: {max_count}, min: {min_count})")
            else:
                print("  ✅ Balanced distribution achieved")

    def search_reddit_posts(self, query_list: List[str]) -> List[Dict[str, str]]:
        """Enhanced search with bias prevention and diverse result collection"""
        print("\n=== SEARCHING FOR REDDIT POSTS (BIAS-FREE) ===")

        found_posts = []
        seen_urls = set()
        
        # Track diversity metrics
        posts_per_query = {}
        subreddit_diversity = {}

        # Target subreddits for gift recommendations (higher relevance scoring)
        TARGET_SUBREDDITS = [
            'giftideas', 'gifts', 'askreddit', 'malefashionadvice',
            'buyitforlife', 'gadgets', 'movies', 'documentaries',
            'sports', 'birthday', 'brothers', 'mensrights', 'dads'
        ]

        # Enhanced relevance keywords with weights
        HIGH_VALUE_KEYWORDS = ['gift', 'present', 'birthday', 'recommendation', 'ideas']
        MEDIUM_VALUE_KEYWORDS = ['suggest', 'buy', 'brother', 'male', 'men', 'guy', 'best']

        # Exclusion keywords for irrelevant posts
        EXCLUDE_KEYWORDS = [
            'aita', 'am i the asshole', 'tifu', 'legal advice', 'relationship advice',
            'breakup', 'divorce', 'drama', 'politics', 'confession'
        ]

        # Process queries with diversity tracking
        for i, query in enumerate(query_list, 1):
            print(f"\n--- Search {i}/{len(query_list)}: {query} ---")
            posts_per_query[query] = 0

            try:
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.GOOGLE_API_KEY,
                    'cx': self.SEARCH_ENGINE_ID,
                    'q': query,
                    'num': 10
                }

                response = requests.get(search_url, params=params, timeout=15)
                print(f"  Response status: {response.status_code}")

                if response.status_code == 403:
                    print("  ✗ API quota exceeded or invalid API key")
                    continue
                elif response.status_code != 200:
                    print(f"  ✗ HTTP Error: {response.status_code}")
                    continue

                search_results = response.json()

                if 'error' in search_results:
                    print(f"  ✗ API Error: {search_results['error']}")
                    continue

                if 'items' not in search_results:
                    print("  ✗ No results found")
                    continue

                total_results = len(search_results['items'])
                print(f"  Found {total_results} total results")

                reddit_count = 0
                relevant_count = 0

                for item in search_results['items']:
                    url = item['link']
                    title = item['title'].lower()
                    snippet = item.get('snippet', '').lower()

                    # Must be Reddit URL
                    if 'reddit.com' not in url.lower():
                        continue

                    reddit_count += 1
                    print(f"    ✓ Reddit URL: {item['title'][:50]}...")

                    # Must be a post with comments (not subreddit home page)
                    if '/comments/' not in url:
                        print(f"      ⚠️  Not a post (no /comments/ in URL)")
                        continue

                    # Calculate relevance score
                    relevance_score = 0

                    # Check subreddit relevance (high weight)
                    subreddit_match = None
                    for sub in TARGET_SUBREDDITS:
                        if f'/r/{sub}/' in url.lower():
                            relevance_score += 5
                            subreddit_match = sub
                            # Track subreddit diversity
                            subreddit_diversity[sub] = subreddit_diversity.get(sub, 0) + 1
                            break

                    # Check for high-value keywords in title (very high weight)
                    title_high_keywords = sum(2 for keyword in HIGH_VALUE_KEYWORDS if keyword in title)
                    relevance_score += title_high_keywords

                    # Check for high-value keywords in snippet
                    snippet_high_keywords = sum(1 for keyword in HIGH_VALUE_KEYWORDS if keyword in snippet)
                    relevance_score += snippet_high_keywords

                    # Check for medium-value keywords
                    medium_keywords = sum(1 for keyword in MEDIUM_VALUE_KEYWORDS if keyword in title or keyword in snippet)
                    relevance_score += medium_keywords * 0.5

                    # Bonus for exact phrase matches
                    if 'gift ideas' in title or 'gift recommendations' in title:
                        relevance_score += 3
                    if 'birthday gift' in title:
                        relevance_score += 2

                    # Penalty for exclusion keywords
                    exclude_penalty = sum(3 for keyword in EXCLUDE_KEYWORDS if keyword in title or keyword in snippet)
                    relevance_score -= exclude_penalty

                    print(f"      Relevance score: {relevance_score:.1f}")
                    if subreddit_match:
                        print(f"      Subreddit: r/{subreddit_match}")

                    # Only include if relevance score is high enough and not already seen
                    if relevance_score >= 2.0 and url not in seen_urls:
                        post_info = {
                            'title': item['title'],
                            'url': url,
                            'snippet': item.get('snippet', ''),
                            'source_query': query,
                            'relevance_score': relevance_score,
                            'subreddit': subreddit_match
                        }

                        found_posts.append(post_info)
                        seen_urls.add(url)
                        relevant_count += 1
                        posts_per_query[query] += 1

                        print(f"        ✅ Added relevant post (Score: {relevance_score:.1f})")
                    else:
                        if relevance_score < 2.0:
                            print(f"        ⚠️  Score too low ({relevance_score:.1f})")
                        else:
                            print(f"        ⚠️  Already seen this URL")

                print(f"  Reddit URLs in results: {reddit_count}/{total_results}")
                print(f"  Relevant posts found: {relevant_count}")
                print(f"  Total collected so far: {len(found_posts)}")

                # Rate limiting
                time.sleep(1)

                # Continue searching to ensure diversity, don't stop early
                if len(found_posts) >= 12:  # Increased threshold for better diversity
                    print("  Good diversity achieved, stopping search")
                    break

            except requests.exceptions.Timeout:
                print(f"  ✗ Timeout error with query '{query}'")
                continue
            except Exception as e:
                print(f"  ✗ Error with query '{query}': {e}")
                continue

        # Print diversity metrics
        print(f"\n📊 SEARCH DIVERSITY METRICS:")
        print(f"  Total unique posts: {len(found_posts)}")
        print(f"  Posts per query: {dict(posts_per_query)}")
        print(f"  Subreddit distribution: {dict(subreddit_diversity)}")

        # Sort by relevance score and return top results
        found_posts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        final_posts = found_posts[:8]  # Top 8 most relevant (increased from 6)

        print(f"\n✅ BIAS-FREE SEARCH COMPLETED - FOUND {len(final_posts)} DIVERSE REDDIT POSTS")
        return final_posts

    def generate_clarifying_questions(self, user_query: str) -> List[str]:
        """Generate clarifying questions, analyzing what's already provided"""
        
        # First check if budget is already mentioned
        budget_check = self._regex_budget_extraction(user_query)
        has_budget_in_query = budget_check['has_budget']
        
        analysis_prompt = f"""
        Analyze the following user query for gift recommendations and determine what information is MISSING.

        USER QUERY: "{user_query}"

        Budget Status: {"Budget mentioned" if has_budget_in_query else "No budget mentioned"}

        First, identify what information is ALREADY PROVIDED in the query:
        - Recipient details (age, gender, relationship, interests)
        - Budget information (${budget_check['display'] if has_budget_in_query else "Not mentioned"})
        - Occasion details
        - Specific preferences or constraints

        Then, generate ONLY 2-4 questions for information that is MISSING or UNCLEAR. Do not ask about information that is already clearly stated in the query.

        Focus on gathering the most important missing information for:
        1. The recipient's specific interests, hobbies, and preferences (if not mentioned)
        2. Budget range - ALWAYS ask if not clearly mentioned (be specific: ask for amount in rupees)
        3. The occasion and relationship context (if not clear)
        4. Any specific constraints or requirements (if not mentioned)

        BUDGET QUESTION RULES:
        - If no budget mentioned, ask: "What's your budget for this gift? (Please specify amount in rupees, e.g., ₹1500, ₹3000, under ₹2000)"
        - If budget is vague, ask for clarification with specific amounts
        - Always ask for budget in rupees with examples

        Generate exactly 2-4 specific, helpful questions ONLY for missing information.

        Format your response as:
        1. [Question about missing information]
        2. [Question about missing information]
        3. [Question about missing information]
        4. [Question about missing information]

        If the user query already contains comprehensive information, generate fewer questions or ask for clarification on ambiguous points only.
        """

        try:
            print("🤖 Analyzing user query for missing information...")
            print(f"Budget in query: {has_budget_in_query} - {budget_check['display'] if has_budget_in_query else 'None'}")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=300,
                temperature=0.7
            )

            questions_text = response.choices[0].message.content.strip()

            # Parse questions into a list
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                if line and (line.startswith(('1.', '2.', '3.', '4.')) or line.endswith('?')):
                    # Remove numbering if present
                    question = line
                    if question[0].isdigit() and '.' in question:
                        question = question.split('.', 1)[1].strip()
                    questions.append(question)

            # Ensure budget question is included if not found in original query
            if not has_budget_in_query:
                budget_question_found = any('budget' in q.lower() for q in questions)
                if not budget_question_found:
                    questions.insert(0, "What's your budget for this gift? (Please specify amount in rupees, e.g., ₹1500, ₹3000, under ₹2000)")

            print("✅ CLARIFYING QUESTIONS GENERATED:")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")

            return questions

        except Exception as e:
            print(f"✗ Error generating clarifying questions: {e}")
            # Fallback questions with explicit budget question
            fallback_questions = [
                "What's your budget for this gift? (Please specify amount in rupees, e.g., ₹1500, ₹3000, under ₹2000)",
                "What are their main interests or hobbies?",
                "What's the occasion for this gift?"
            ]
            print("✓ Using fallback questions")
            return fallback_questions

    def analyze_user_response_for_skip(self, current_question: str, user_answer: str, remaining_questions: List[str]) -> List[str]:
        """Analyze user response to see if any remaining questions can be skipped"""
        if not remaining_questions:
            return []
        
        analysis_prompt = f"""
        The user was asked: "{current_question}"
        The user answered: "{user_answer}"

        Remaining questions to ask:
        {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(remaining_questions)])}

        Based on the user's answer, determine which of the remaining questions are now UNNECESSARY to ask because:
        1. The user's answer already provided that information
        2. The user's answer makes the question irrelevant
        3. The information can be reasonably inferred from their response

        Return only the questions that should STILL BE ASKED, maintaining their original numbering.

        If all remaining questions should still be asked, return all of them.
        If some can be skipped, return only the necessary ones.

        Format: Return only the questions that should still be asked, one per line, without numbering.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=200,
                temperature=0.3
            )

            filtered_text = response.choices[0].message.content.strip()
            
            # Parse the filtered questions
            filtered_questions = []
            for line in filtered_text.split('\n'):
                line = line.strip()
                if line and line.endswith('?'):
                    # Clean up any remaining numbering
                    if line[0].isdigit() and '.' in line:
                        line = line.split('.', 1)[1].strip()
                    filtered_questions.append(line)
            
            # If parsing fails, return original questions
            if not filtered_questions:
                return remaining_questions
                
            print(f"✓ Filtered questions: {len(remaining_questions)} -> {len(filtered_questions)}")
            return filtered_questions

        except Exception as e:
            print(f"✗ Error filtering questions: {e}")
            return remaining_questions

    def create_enhanced_query(self, original_query: str, clarifying_answers: Dict[str, str]) -> str:
        """Create enhanced search context with debug logging"""
        print("\n=== CREATING ENHANCED SEARCH CONTEXT ===")

        # Prepare enhanced context
        enhanced_context = f"Original request: {original_query}\n\n"
        enhanced_context += "Additional details:\n"

        for answer_data in clarifying_answers.values():
            enhanced_context += f"- {answer_data['question']} {answer_data['answer']}\n"

        print("✓ Enhanced context created:")
        print(f"  Original query: {original_query[:50]}...")
        print(f"  Additional details: {len(clarifying_answers)} answers collected")
        print(f"  Full enhanced context: {enhanced_context}")

        return enhanced_context

    def extract_reddit_content(self, reddit_posts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Extract Reddit content from posts"""
        print("\n=== EXTRACTING REDDIT CONTENT ===")

        if not reddit_posts:
            print("❌ No Reddit posts to extract content from")
            return []

        extracted_content = []

        for i, post in enumerate(reddit_posts, 1):
            print(f"\n--- Processing Reddit Post {i}/{len(reddit_posts)} ---")
            print(f"Title: {post['title'][:60]}...")
            print(f"URL: {post['url']}")

            try:
                # Extract post ID from Reddit URL
                if '/comments/' in post['url']:
                    # URL format: https://www.reddit.com/r/subreddit/comments/POST_ID/title/
                    url_parts = post['url'].split('/comments/')
                    if len(url_parts) > 1:
                        post_id = url_parts[1].split('/')[0]
                    else:
                        print(f"  ✗ Could not extract post ID from URL")
                        continue
                else:
                    print(f"  ✗ URL doesn't contain '/comments/' - skipping")
                    continue

                print(f"  ✓ Extracted post ID: {post_id}")

                # Get submission using Reddit API
                submission = self.reddit.submission(id=post_id)

                # Extract post information
                post_data = {
                    'title': submission.title,
                    'selftext': submission.selftext if hasattr(submission, 'selftext') else '',
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'url': post['url'],
                    'subreddit': str(submission.subreddit),
                    'author': str(submission.author) if submission.author else 'Unknown',
                    'comments': []
                }

                print(f"  ✓ Post title: {submission.title[:50]}...")
                print(f"  ✓ Post score: {submission.score}")
                print(f"  ✓ Subreddit: r/{submission.subreddit}")
                print(f"  ✓ Number of comments: {submission.num_comments}")

                # Extract top comments (limit to 50)
                print(f"  🔍 Extracting top 50 comments...")

                # Replace "more comments" objects to get actual comments
                submission.comments.replace_more(limit=0)

                comment_count = 0
                for comment in submission.comments.list():
                    if comment_count >= 50:  # Limit to top 50 comments
                        break

                    if hasattr(comment, 'body') and comment.body and comment.body != '[deleted]' and comment.body != '[removed]':
                        comment_data = {
                            'body': comment.body,
                            'score': comment.score if hasattr(comment, 'score') else 0,
                            'author': str(comment.author) if comment.author else 'Unknown'
                        }

                        post_data['comments'].append(comment_data)
                        comment_count += 1

                print(f"  ✅ Extracted {comment_count} valid comments")
                extracted_content.append(post_data)

                # Rate limiting to respect Reddit API
                time.sleep(1)

            except Exception as e:
                print(f"  ✗ Error extracting content from {post['url']}: {e}")
                continue

        print(f"\n✅ CONTENT EXTRACTION COMPLETED")
        print(f"Successfully processed {len(extracted_content)} out of {len(reddit_posts)} posts")

        return extracted_content

    def _extract_budget_from_context(self, enhanced_context: str) -> Dict[str, Any]:
        """Extract budget information from the enhanced context with enhanced parsing"""
        print("\n=== EXTRACTING BUDGET INFORMATION ===")
        print(f"Context to analyze: {enhanced_context}")
        
        # First try regex-based extraction for common patterns
        budget_info = self._regex_budget_extraction(enhanced_context)
        if budget_info['has_budget']:
            print(f"✅ Regex extracted budget: {budget_info['display']}")
            return budget_info
        
        # Fallback to AI extraction
        budget_prompt = f"""
        CRITICAL BUDGET EXTRACTION TASK - Find ANY mention of money/budget in this text:

        USER CONTEXT:
        {enhanced_context}

        Look for ANY of these patterns:
        - Numbers followed by "Rs", "rupees", "₹"  
        - Budget mentions like "1500 Rs", "₹2000", "2500 rupees"
        - Range mentions like "1000-2000", "₹1500 to ₹3000"
        - Limit mentions like "under 2000", "below ₹1500", "maximum 3000"
        - Around mentions like "around 1500", "approximately ₹2000"

        EXAMPLES:
        - "1500 Rs" -> max_amount = 1500, display = "₹1500"
        - "under 2000 rupees" -> max_amount = 2000, display = "under ₹2000" 
        - "₹1000 to ₹3000" -> min_amount = 1000, max_amount = 3000, display = "₹1000-₹3000"
        - "around ₹2500" -> max_amount = 2500, display = "around ₹2500"
        - "budget of 1800" -> max_amount = 1800, display = "₹1800"

        Return in this EXACT JSON format:
        {{
            "min_amount": [minimum budget in rupees as integer, or 0],
            "max_amount": [maximum budget in rupees as integer],
            "currency": "INR",
            "display": "[how user mentioned it, e.g., '₹1500', 'under ₹2000']",
            "has_budget": [true if ANY budget number found, false if none],
            "is_strict": [true if "under", "below", "within" mentioned],
            "budget_keywords": ["found budget-related words"]
        }}

        CRITICAL: If you find ANY number with Rs/rupees/₹, set has_budget = true and extract that amount.
        If NO budget mentioned at all, set has_budget = false and max_amount = 10000.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": budget_prompt}],
                max_tokens=300,
                temperature=0.1
            )

            response_text = response.choices[0].message.content.strip()
            print(f"AI response: {response_text}")
            
            import json
            budget_info = json.loads(response_text)
            
            # Validation and fallback
            if not budget_info.get('has_budget', False) or budget_info.get('max_amount', 0) <= 0:
                print("⚠️ No valid budget found, using default")
                budget_info = {
                    "min_amount": 0,
                    "max_amount": 10000,
                    "currency": "INR", 
                    "display": "₹0-10000 (default)",
                    "has_budget": False,
                    "is_strict": False,
                    "budget_keywords": ["affordable"]
                }
            
            print(f"✅ AI extracted budget: {budget_info['display']} (Max: ₹{budget_info['max_amount']})")
            return budget_info
            
        except Exception as e:
            print(f"✗ Error in AI budget extraction: {e}")
            # Fallback budget
            return {
                "min_amount": 0,
                "max_amount": 10000,
                "currency": "INR", 
                "display": "₹0-10000 (default)",
                "has_budget": False,
                "is_strict": False,
                "budget_keywords": ["affordable"]
            }

    def _regex_budget_extraction(self, text: str) -> Dict[str, Any]:
        """Extract budget using regex patterns for common formats"""
        import re
        
        text_lower = text.lower()
        print(f"Regex analyzing: {text_lower}")
        
        # Pattern 1: Direct amount mentions like "1500 Rs", "₹2000", "2500 rupees"
        patterns = [
            r'₹\s*(\d+(?:,\d+)*)',  # ₹1500, ₹2,000
            r'(\d+(?:,\d+)*)\s*rs\.?',  # 1500 Rs, 2000 rs
            r'(\d+(?:,\d+)*)\s*rupees?',  # 1500 rupees, 2000 rupee
            r'budget.*?(\d+(?:,\d+)*)',  # budget of 1500, budget 2000
            r'around.*?(\d+(?:,\d+)*)',  # around 1500
            r'maximum.*?(\d+(?:,\d+)*)',  # maximum 2000
            r'under.*?(\d+(?:,\d+)*)',  # under 1500
            r'below.*?(\d+(?:,\d+)*)',  # below 2000
            r'within.*?(\d+(?:,\d+)*)',  # within 1800
            r'up\s*to.*?(\d+(?:,\d+)*)',  # up to 2500
        ]
        
        extracted_amounts = []
        budget_keywords = []
        is_strict = False
        
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    amount = int(match.replace(',', ''))
                    if 100 <= amount <= 100000:  # Reasonable budget range
                        extracted_amounts.append(amount)
                        print(f"Found amount: {amount} with pattern: {pattern}")
                except ValueError:
                    continue
        
        # Check for strict keywords
        strict_keywords = ['under', 'below', 'within', 'maximum', 'max', 'up to']
        for keyword in strict_keywords:
            if keyword in text_lower:
                is_strict = True
                budget_keywords.append(keyword)
        
        # Check for range patterns like "1000-2000", "₹1500 to ₹3000"
        range_patterns = [
            r'₹?\s*(\d+(?:,\d+)*)\s*(?:to|-)\s*₹?\s*(\d+(?:,\d+)*)',
            r'between.*?(\d+(?:,\d+)*).*?and.*?(\d+(?:,\d+)*)',
        ]
        
        min_amount = 0
        max_amount = 0
        
        for pattern in range_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    min_val = int(match[0].replace(',', ''))
                    max_val = int(match[1].replace(',', ''))
                    if 100 <= min_val <= max_val <= 100000:
                        min_amount = min_val
                        max_amount = max_val
                        print(f"Found range: {min_val}-{max_val}")
                        break
                except (ValueError, IndexError):
                    continue
        
        # If no range found, use single amounts
        if not max_amount and extracted_amounts:
            max_amount = max(extracted_amounts)
            if len(extracted_amounts) > 1:
                min_amount = min(extracted_amounts)
        
        if max_amount > 0:
            # Create display string
            if min_amount > 0 and min_amount != max_amount:
                display = f"₹{min_amount}-₹{max_amount}"
            elif is_strict:
                display = f"under ₹{max_amount}"
            else:
                display = f"₹{max_amount}"
            
            return {
                "min_amount": min_amount,
                "max_amount": max_amount,
                "currency": "INR",
                "display": display,
                "has_budget": True,
                "is_strict": is_strict,
                "budget_keywords": budget_keywords
            }
        
        print("No budget found in regex extraction")
        return {
            "min_amount": 0,
            "max_amount": 0,
            "currency": "INR",
            "display": "",
            "has_budget": False,
            "is_strict": False,
            "budget_keywords": []
        }

    def _check_budget_compliance(self, snippet: str, title: str, budget_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check if product is within budget with strict price extraction"""
        text = (snippet + " " + title).lower()
        max_budget = budget_info['max_amount']
        min_budget = budget_info.get('min_amount', 0)
        
        within_budget = False
        extracted_prices = []
        confidence_score = 0
        
        # Enhanced price patterns for Indian currency
        price_patterns = [
            r'₹\s*(\d{1,2}(?:,\d{2,3})*)',  # ₹1,000 or ₹500
            r'rs\.?\s*(\d{1,2}(?:,\d{2,3})*)',  # Rs. 1000 or rs 500
            r'inr\s*(\d{1,2}(?:,\d{2,3})*)',  # INR 1000
            r'rupees?\s*(\d{1,2}(?:,\d{2,3})*)',  # rupees 1000
            r'price.*?₹\s*(\d{1,2}(?:,\d{2,3})*)',  # price ₹1000
            r'cost.*?₹\s*(\d{1,2}(?:,\d{2,3})*)',  # cost ₹1000
            r'only.*?₹\s*(\d{1,2}(?:,\d{2,3})*)',  # only ₹1000
            r'starting.*?₹\s*(\d{1,2}(?:,\d{2,3})*)',  # starting ₹1000
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Remove commas and convert to int
                    price = int(match.replace(',', ''))
                    extracted_prices.append(price)
                    
                    # Check if within budget
                    if min_budget <= price <= max_budget:
                        within_budget = True
                        confidence_score += 5  # High confidence for within budget
                    elif price <= max_budget * 1.1:  # Within 10% of budget
                        confidence_score += 2  # Medium confidence
                        
                except ValueError:
                    continue
        
        # Look for budget-friendly keywords
        budget_keywords = [
            f"under ₹{max_budget}", f"below ₹{max_budget}", f"within ₹{max_budget}",
            "affordable", "cheap", "budget", "low price", "discount", "sale"
        ]
        
        for keyword in budget_keywords:
            if keyword in text:
                confidence_score += 2
                if not extracted_prices:  # If no price found but budget keywords present
                    within_budget = True
        
        # Look for expensive keywords (negative indicators)
        expensive_keywords = [
            f"above ₹{max_budget}", f"over ₹{max_budget}", f"more than ₹{max_budget}",
            "expensive", "premium", "luxury", "high-end"
        ]
        
        for keyword in expensive_keywords:
            if keyword in text:
                confidence_score -= 3
                within_budget = False
        
        return {
            'within_budget': within_budget,
            'extracted_prices': extracted_prices,
            'confidence_score': max(confidence_score, 0),
            'has_price_info': len(extracted_prices) > 0
        }

    def _get_budget_filtered_indian_results(self, product_name: str, budget_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get search results from Indian sites with budget filtering"""
        raw_results = []
        seen_urls = set()
        max_budget = budget_info['max_amount']

        # Indian e-commerce sites
        INDIAN_SITES = [
            'amazon.in', 'flipkart.com', 'myntra.com', 'ajio.com',
            'tatacliq.com', 'snapdeal.com', 'nykaa.com', 'meesho.com'
        ]

        # Budget-focused search queries
        queries = [
            f"{product_name} under ₹{max_budget} site:amazon.in",
            f"{product_name} under ₹{max_budget} site:flipkart.com",
            f"{product_name} budget ₹{max_budget} site:myntra.com",
            f"{product_name} affordable under ₹{max_budget} India",
            f"{product_name} cheap under ₹{max_budget} Indian sites",
            f"buy {product_name} ₹{max_budget} India online shopping"
        ]

        print(f"  🔍 Searching with budget filter: ₹{max_budget}")

        for query in queries:
            if len(raw_results) >= 20:
                break
                
            try:
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.GOOGLE_API_KEY,
                    'cx': self.SEARCH_ENGINE_ID,
                    'q': query,
                    'num': 10,
                    'gl': 'in',
                    'hl': 'en'
                }

                response = requests.get(search_url, params=params, timeout=15)
                
                if response.status_code != 200:
                    continue

                search_results = response.json()
                
                if 'items' not in search_results:
                    continue

                for item in search_results['items']:
                    url = item['link'].lower()
                    
                    if url in seen_urls:
                        continue
                    
                    # Check if from Indian site
                    is_indian_site = any(site in url for site in INDIAN_SITES)
                    if not is_indian_site:
                        continue
                    
                    title = item['title']
                    snippet = item.get('snippet', '')
                    
                    # Budget compliance check
                    budget_check = self._check_budget_compliance(snippet, title, budget_info)
                    
                    # Only include if within budget or likely within budget
                    if budget_check['within_budget'] or budget_check['confidence_score'] >= 3:
                        raw_results.append({
                            'title': title,
                            'url': item['link'],
                            'snippet': snippet,
                            'site': next((site for site in INDIAN_SITES if site in url), 'unknown'),
                            'budget_check': budget_check
                        })
                        
                        seen_urls.add(url)
                        budget_status = "✅" if budget_check['within_budget'] else "🟡"
                        print(f"    {budget_status} {title[:50]}... (Score: {budget_check['confidence_score']})")
                
                time.sleep(1)
                    
            except Exception as e:
                print(f"    ✗ Error with query: {e}")
                continue

        # Sort by budget relevance
        raw_results.sort(key=lambda x: x['budget_check']['confidence_score'], reverse=True)
        
        print(f"  📊 Found {len(raw_results)} budget-filtered results")
        return raw_results

    def search_products_to_buy(self, selected_products: List[str], enhanced_context: str) -> List[Dict[str, Any]]:
        """Search for products with strict budget enforcement"""
        print("\n=== SEARCHING PRODUCTS WITH BUDGET FILTERING ===")

        # Extract budget information
        budget_info = self._extract_budget_from_context(enhanced_context)
        print(f"  💰 Budget limit: ₹{budget_info['max_amount']}")

        all_product_results = []

        for product in selected_products:
            # Extract clean product name
            product_name = product.split(' - ')[0] if ' - ' in product else product
            if '. ' in product_name:
                product_name = product_name.split('. ', 1)[1]
            product_name = product_name.replace('**', '').strip()

            print(f"\n--- Budget Search: {product_name} (Max: ₹{budget_info['max_amount']}) ---")

            # Get budget-filtered results
            filtered_results = self._get_budget_filtered_indian_results(product_name, budget_info)
            
            if not filtered_results:
                print(f"  ❌ No products found within ₹{budget_info['max_amount']}")
                all_product_results.append({
                    'category': product_name,
                    'original_recommendation': product,
                    'products': [],
                    'budget_info': budget_info
                })
                continue

            # Take top 3 budget-compliant results
            final_products = filtered_results[:3]
            
            all_product_results.append({
                'category': product_name,
                'original_recommendation': product,
                'products': final_products,
                'budget_info': budget_info
            })

            print(f"  ✅ Found {len(final_products)} budget-compliant products")

        return all_product_results


# Initialize the recommendation system
try:
    gift_system = GiftRecommendationSystem()
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("Please check your environment variables are set correctly.")
    exit(1)

# Session storage for conversation state
conversation_sessions = {}

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/start_conversation', methods=['POST'])
def start_conversation():
    """Start a new conversation"""
    session_id = str(uuid.uuid4())
    conversation_sessions[session_id] = {
        'step': 1,
        'user_query': '',
        'clarifying_questions': [],
        'clarifying_answers': {},
        'recommendations': [],
        'selected_products': []
    }
    
    return jsonify({
        'session_id': session_id,
        'message': 'Hi! Ready to find the perfect gift? 🎁'
    })

@app.route('/api/process_message', methods=['POST'])
def process_message():
    """Process user messages with comprehensive logging"""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        if session_data['step'] == 1:
            # Log initial user query
            interaction_logger_instance.log_user_query(session_id, message)
            
            # Initial query - USE REAL FUNCTION
            session_data['user_query'] = message
            
            # REAL API CALL - not mock
            questions = gift_system.generate_clarifying_questions(message)
            session_data['clarifying_questions'] = questions
            session_data['step'] = 2
            
            # Log generated clarifying questions
            interaction_logger_instance.log_clarifying_questions(session_id, questions)
            
            # Return first question directly without showing the list
            return jsonify({
                'type': 'first_question',
                'question': questions[0] if questions else "What's your budget for this gift?",
                'total_questions': len(questions)
            })
            
        elif session_data['step'] == 2:
            # Clarifying answers with intelligent skipping
            current_q_index = len(session_data['clarifying_answers'])
            current_question = session_data['clarifying_questions'][current_q_index]
            
            # Log user's answer
            interaction_logger_instance.log_user_answer(session_id, current_q_index, current_question, message)
            
            # Store the current answer
            session_data['clarifying_answers'][f'question_{current_q_index + 1}'] = {
                'question': current_question,
                'answer': message
            }
            
            # Check if there are more questions
            if current_q_index + 1 < len(session_data['clarifying_questions']):
                # Get remaining questions
                remaining_questions = session_data['clarifying_questions'][current_q_index + 1:]
                
                # Analyze if any can be skipped based on user's response
                filtered_questions = gift_system.analyze_user_response_for_skip(
                    current_question, message, remaining_questions
                )
                
                if filtered_questions:
                    # Update the clarifying_questions list with filtered questions
                    session_data['clarifying_questions'] = (
                        session_data['clarifying_questions'][:current_q_index + 1] + 
                        filtered_questions
                    )
                    
                    # Ask the next question directly
                    return jsonify({
                        'type': 'next_question',
                        'question': filtered_questions[0],
                        'current_question': current_q_index + 1
                    })
                else:
                    # All remaining questions can be skipped, start analysis
                    session_data['step'] = 3
                    return jsonify({
                        'type': 'start_analysis',
                        'message': 'Got it! Let me find some great options for you...'
                    })
            else:
                # All questions answered, start analysis
                session_data['step'] = 3
                return jsonify({
                    'type': 'start_analysis',
                    'message': 'Perfect! Let me find the best recommendations...'
                })
        
        else:
            return jsonify({'error': 'Invalid step'}), 400
            
    except Exception as e:
        interaction_logger_instance.log_error(session_id, 'PROCESS_MESSAGE_ERROR', str(e), session_data.get('step', 'unknown'))
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/api/analyze_requirements', methods=['POST'])
def analyze_requirements():
    """Analyze user requirements with comprehensive logging"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        print("\n🚀 STARTING BALANCED ANALYSIS WITH LOGGING")
        
        # STEP 1: Create enhanced context
        enhanced_context = gift_system.create_enhanced_query(
            session_data['user_query'], 
            session_data['clarifying_answers']
        )
        
        # Log enhanced context creation
        interaction_logger_instance.log_enhanced_context(session_id, enhanced_context)
        
        # STEP 2: Extract preferences for balanced analysis
        preferences = gift_system.extract_preferences_and_interests(enhanced_context)
        
        # Log extracted preferences
        interaction_logger_instance.log_preferences_extracted(session_id, preferences)
        
        # STEP 3: Generate BALANCED search queries
        print("📋 Generating balanced search queries...")
        optimized_queries = gift_system.optimize_search_query(enhanced_context)
        
        # Log optimized queries
        interaction_logger_instance.log_optimized_queries(session_id, optimized_queries)
        
        if not optimized_queries:
            raise Exception("Failed to generate search queries")
        
        # STEP 4: Search Reddit posts with bias prevention
        print("🔍 Searching Reddit posts (bias-free)...")
        reddit_posts = gift_system.search_reddit_posts(optimized_queries)
        
        if not reddit_posts:
            raise Exception("No Reddit posts found")
        
        # STEP 5: Extract Reddit content
        print("📖 Extracting Reddit content...")
        reddit_content = gift_system.extract_reddit_content(reddit_posts)
        
        # Log Reddit search results
        interaction_logger_instance.log_reddit_search_results(session_id, reddit_posts, reddit_content)
        
        if not reddit_content:
            raise Exception("No Reddit content extracted")
        
        # STEP 6: BALANCED analysis for product recommendations
        print("🤖 Performing balanced content analysis...")
        recommendations = gift_system.analyze_for_product_recommendations(reddit_content, enhanced_context)
        
        if not recommendations:
            raise Exception("No recommendations generated")
        
        # Clean up ** stars from product names
        cleaned_recommendations = []
        for rec in recommendations:
            cleaned_rec = rec.replace('**', '')
            cleaned_recommendations.append(cleaned_rec)
        
        # Log product recommendations
        interaction_logger_instance.log_product_recommendations(session_id, cleaned_recommendations)
        
        # Store recommendations in session
        session_data['recommendations'] = cleaned_recommendations
        session_data['reddit_content'] = reddit_content
        session_data['enhanced_context'] = enhanced_context
        session_data['step'] = 4
        
        print(f"✅ BALANCED ANALYSIS COMPLETED - {len(cleaned_recommendations)} recommendations generated")
        
        return jsonify({
            'type': 'recommendations',
            'recommendations': cleaned_recommendations
        })
        
    except Exception as e:
        interaction_logger_instance.log_error(session_id, 'ANALYZE_REQUIREMENTS_ERROR', str(e), session_data.get('step', 'unknown'))
        print(f"❌ Error in balanced analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/select_products', methods=['POST'])
def select_products():
    """Handle product selection with comprehensive logging"""
    data = request.json
    session_id = data.get('session_id')
    selected_indices = data.get('selected_indices')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        print(f"\n🎯 PROCESSING PRODUCT SELECTION WITH LOGGING")
        
        # Get selected products from stored recommendations
        selected_products = [session_data['recommendations'][i] for i in selected_indices]
        session_data['selected_products'] = selected_products
        
        # Log user selection
        interaction_logger_instance.log_user_selection(session_id, selected_indices, selected_products)
        
        print("✅ Selected products:")
        for i, product in enumerate(selected_products, 1):
            print(f"  {i}. {product[:100]}...")
        
        # Extract budget information and log it
        budget_info = gift_system._extract_budget_from_context(session_data['enhanced_context'])
        interaction_logger_instance.log_budget_extraction(session_id, budget_info)
        
        # Search with budget filtering
        print("🛒 Searching for budget-compliant products...")
        product_results = gift_system.search_products_to_buy(
            selected_products, 
            session_data['enhanced_context']
        )
        
        # Log final product results
        interaction_logger_instance.log_final_products(session_id, product_results)
        
        print(f"✅ BUDGET-FILTERED SEARCH COMPLETED")
        
        return jsonify({
            'type': 'final_recommendations',
            'product_results': product_results
        })
        
    except Exception as e:
        interaction_logger_instance.log_error(session_id, 'SELECT_PRODUCTS_ERROR', str(e), session_data.get('step', 'unknown'))
        print(f"❌ Error in product selection: {e}")
        return jsonify({'error': f'Product selection failed: {str(e)}'}), 500

@app.route('/api/test_apis', methods=['GET'])
def test_apis():
    """Test API connectivity"""
    results = {}
    
    # Test OpenAI
    try:
        response = gift_system.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'API test successful'"}],
            max_tokens=10
        )
        results['openai'] = '✅ Working'
    except Exception as e:
        results['openai'] = f'❌ Failed: {str(e)}'
    
    # Test Google Search
    try:
        test_url = "https://www.googleapis.com/customsearch/v1"
        test_params = {
            'key': gift_system.GOOGLE_API_KEY,
            'cx': gift_system.SEARCH_ENGINE_ID,
            'q': 'test',
            'num': 1
        }
        response = requests.get(test_url, params=test_params, timeout=10)
        if response.status_code == 200:
            results['google_search'] = '✅ Working'
        else:
            results['google_search'] = f'❌ Status: {response.status_code}'
    except Exception as e:
        results['google_search'] = f'❌ Failed: {str(e)}'
    
    # Test Reddit
    try:
        test_subreddit = gift_system.reddit.subreddit('test')
        if test_subreddit:
            results['reddit'] = '✅ Working'
    except Exception as e:
        results['reddit'] = f'❌ Failed: {str(e)}'
    
    return jsonify(results)

@app.route('/health')
def health_check():
    """Health check endpoint for deployment platforms"""
    return jsonify({'status': 'healthy', 'message': 'Gift Recommendation System is running'})

# Add a new route to view interaction logs (optional - for debugging)
@app.route('/api/interaction_logs/<session_id>')
def get_interaction_logs(session_id):
    """Get interaction logs for a specific session (for debugging)"""
    try:
        logs = []
        with open('user_interactions.log', 'r') as f:
            for line in f:
                if f'"session_id": "{session_id}"' in line:
                    # Extract the JSON part from the log line
                    json_start = line.find('USER_INTERACTION: ') + len('USER_INTERACTION: ')
                    json_data = line[json_start:].strip()
                    logs.append(json.loads(json_data))
        
        return jsonify({
            'session_id': session_id,
            'logs': logs,
            'total_events': len(logs)
        })
    except Exception as e:
        return jsonify({'error': f'Failed to retrieve logs: {str(e)}'}), 500

# Add a route to get summary statistics
@app.route('/api/interaction_summary')
def get_interaction_summary():
    """Get summary of all user interactions"""
    try:
        stats = {
            'total_sessions': 0,
            'total_queries': 0,
            'total_recommendations': 0,
            'total_final_products': 0,
            'error_count': 0,
            'budget_extraction_success_rate': 0,
            'average_budget': 0
        }
        
        sessions_seen = set()
        budget_extractions = []
        
        with open('user_interactions.log', 'r') as f:
            for line in f:
                if 'USER_INTERACTION:' in line:
                    try:
                        json_start = line.find('USER_INTERACTION: ') + len('USER_INTERACTION: ')
                        json_data = line[json_start:].strip()
                        log_entry = json.loads(json_data)
                        
                        session_id = log_entry['session_id']
                        event_type = log_entry['event_type']
                        
                        sessions_seen.add(session_id)
                        
                        if event_type == 'USER_QUERY':
                            stats['total_queries'] += 1
                        elif event_type == 'PRODUCT_RECOMMENDATIONS_GENERATED':
                            stats['total_recommendations'] += log_entry['data']['recommendation_count']
                        elif event_type == 'FINAL_PRODUCTS_FOUND':
                            stats['total_final_products'] += log_entry['data']['total_products']
                        elif event_type == 'BUDGET_EXTRACTED':
                            budget_info = log_entry['data']['budget_info']
                            budget_extractions.append(budget_info)
                        elif event_type == 'ERROR':
                            stats['error_count'] += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        stats['total_sessions'] = len(sessions_seen)
        
        # Calculate budget statistics
        if budget_extractions:
            successful_budgets = [b for b in budget_extractions if b.get('has_budget', False)]
            stats['budget_extraction_success_rate'] = len(successful_budgets) / len(budget_extractions)
            if successful_budgets:
                stats['average_budget'] = sum(b.get('max_amount', 0) for b in successful_budgets) / len(successful_budgets)
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Save the HTML template (same as before, no changes to UI)
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Gift Recommendation System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f7f7f8;
            color: #202123;
            line-height: 1.6;
        }

        .container {
            max-width: 768px;
            margin: 0 auto;
            height: 100vh;
            display: flex;
            flex-direction: column;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }

        .header {
            padding: 20px;
            border-bottom: 1px solid #e5e5e5;
            background-color: white;
            text-align: center;
        }

        .header h1 {
            font-size: 24px;
            font-weight: 600;
            color: #202123;
            margin-bottom: 8px;
        }

        .header p {
            color: #6b7280;
            font-size: 14px;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .message.user .message-content {
            background-color: #10a37f;
            color: white;
            margin-left: 48px;
        }

        .message.assistant .message-content {
            background-color: #f7f7f8;
            color: #202123;
            margin-right: 48px;
        }

        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 12px;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background-color: #10a37f;
            color: white;
        }

        .message.assistant .message-avatar {
            background-color: #19c37d;
            color: white;
        }

        .message-content {
            padding: 12px 16px;
            border-radius: 8px;
            max-width: 100%;
            word-wrap: break-word;
        }

        .step-indicator {
            display: inline-block;
            background-color: #10a37f;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #6b7280;
            font-style: italic;
        }

        .loading::after {
            content: '';
            width: 12px;
            height: 12px;
            border: 2px solid #e5e5e5;
            border-top: 2px solid #10a37f;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .input-container {
            padding: 20px;
            border-top: 1px solid #e5e5e5;
            background-color: white;
        }

        .input-wrapper {
            position: relative;
            background-color: #f7f7f8;
            border-radius: 12px;
            border: 1px solid #e5e5e5;
            padding: 12px 50px 12px 16px;
            transition: border-color 0.2s;
        }

        .input-wrapper:focus-within {
            border-color: #10a37f;
            box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1);
        }

        .input-field {
            width: 100%;
            border: none;
            background: transparent;
            outline: none;
            resize: none;
            font-size: 16px;
            line-height: 1.5;
            color: #202123;
            min-height: 24px;
            max-height: 120px;
        }

        .input-field::placeholder {
            color: #9ca3af;
        }

        .send-button {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            background-color: #10a37f;
            color: white;
            border: none;
            border-radius: 6px;
            width: 32px;
            height: 32px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }

        .send-button:hover:not(:disabled) {
            background-color: #0d8a6b;
        }

        .send-button:disabled {
            background-color: #d1d5db;
            cursor: not-allowed;
        }

        .question-list {
            list-style: none;
            padding: 0;
            margin: 12px 0;
        }

        .question-list li {
            padding: 6px 0;
            color: #374151;
        }

        .question-list li strong {
            color: #10a37f;
        }

        .selection-container {
            background-color: #f9fafb;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 16px;
            margin: 12px 0;
        }

        .selection-title {
            font-weight: 600;
            margin-bottom: 12px;
            color: #202123;
        }

        .selection-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px;
        }

        .selection-item {
            background-color: white;
            border: 2px solid #e5e5e5;
            border-radius: 6px;
            padding: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .selection-item:hover {
            border-color: #10a37f;
            background-color: #f0fdf4;
        }

        .selection-item.selected {
            border-color: #10a37f;
            background-color: #ecfdf5;
        }

        .selection-item-number {
            display: inline-block;
            width: 24px;
            height: 24px;
            background-color: #10a37f;
            color: white;
            border-radius: 50%;
            line-height: 24px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            text-align: center;
        }

        .selection-item-title {
            font-size: 14px;
            font-weight: 500;
            color: #202123;
            margin-bottom: 4px;
        }

        .selection-item-description {
            font-size: 12px;
            color: #6b7280;
        }

        .confirm-button {
            background-color: #10a37f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            margin-top: 16px;
            transition: background-color 0.2s;
        }

        .confirm-button:hover:not(:disabled) {
            background-color: #0d8a6b;
        }

        .confirm-button:disabled {
            background-color: #d1d5db;
            cursor: not-allowed;
        }

        .final-recommendations {
            background-color: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            padding: 20px;
            margin: 12px 0;
        }

        .final-recommendations h3 {
            color: #059669;
            margin-bottom: 16px;
            font-size: 18px;
        }

        .recommendation-category {
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid #d1fae5;
        }

        .recommendation-category:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }

        .category-title {
            font-weight: 600;
            color: #202123;
            margin-bottom: 12px;
            font-size: 16px;
        }

        .product-card {
            background-color: white;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            transition: box-shadow 0.2s;
        }

        .product-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .product-title {
            font-weight: 600;
            color: #202123;
            margin-bottom: 8px;
        }

        .product-description {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 12px;
        }

        .buy-link {
            display: inline-block;
            background-color: #10a37f;
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.2s;
        }

        .buy-link:hover {
            background-color: #0d8a6b;
        }

        .error-message {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 12px;
            border-radius: 8px;
            margin: 12px 0;
        }

        @media (max-width: 640px) {
            .message.user .message-content {
                margin-left: 24px;
            }
            
            .message.assistant .message-content {
                margin-right: 24px;
            }

            .selection-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Gift Recommendation System</h1>
            <p>Get personalized gift recommendations from community discussions (Budget-filtered Indian sites)</p>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="step-indicator">Step 1</div>
                    <p>Hi! I'll help you find the perfect gift! 🎁</p>
                    <p>Tell me what you're looking for and who it's for:</p>
                </div>
            </div>
        </div>

        <div class="input-container">
            <div class="input-wrapper">
                <textarea 
                    class="input-field" 
                    id="messageInput" 
                    placeholder="Type your message here..."
                    rows="1"
                ></textarea>
                <button class="send-button" id="sendButton">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                </button>
            </div>
        </div>
    </div>

    <script>
        class GiftRecommendationUI {
            constructor() {
                this.chatContainer = document.getElementById('chatContainer');
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.sessionId = null;
                
                this.initializeEventListeners();
                this.autoResizeTextarea();
                this.startConversation();
            }

            async startConversation() {
                try {
                    const response = await fetch('/api/start_conversation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });
                    const data = await response.json();
                    this.sessionId = data.session_id;
                    console.log('Conversation started with session ID:', this.sessionId);
                } catch (error) {
                    console.error('Failed to start conversation:', error);
                    this.displayErrorMessage('Failed to connect to server. Please refresh and try again.');
                }
            }

            initializeEventListeners() {
                this.sendButton.addEventListener('click', () => this.handleSend());
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.handleSend();
                    }
                });
                this.messageInput.addEventListener('input', () => this.autoResizeTextarea());
            }

            autoResizeTextarea() {
                this.messageInput.style.height = 'auto';
                this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
            }

            async handleSend() {
                const message = this.messageInput.value.trim();
                if (!message || !this.sessionId) return;

                this.displayUserMessage(message);
                this.messageInput.value = '';
                this.autoResizeTextarea();
                this.setLoading(true);

                try {
                    const response = await fetch('/api/process_message', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: this.sessionId,
                            message: message
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const data = await response.json();
                    await this.handleResponse(data);
                } catch (error) {
                    console.error('Error sending message:', error);
                    this.displayErrorMessage('Failed to process your message. Please try again.');
                } finally {
                    this.setLoading(false);
                }
            }

            async handleResponse(data) {
                if (data.error) {
                    this.displayErrorMessage(data.error);
                    return;
                }

                switch (data.type) {
                    case 'first_question':
                        this.displayFirstQuestion(data.question);
                        break;
                    case 'next_question':
                        this.displayNextQuestion(data.question, data.current_question);
                        break;
                    case 'start_analysis':
                        await this.startAnalysis();
                        break;
                    default:
                        this.displayAssistantMessage(data.message || 'Unexpected response');
                }
            }

            displayFirstQuestion(question) {
                const questionHtml = `
                    <div class="step-indicator">Step 2</div>
                    <p><strong>${question}</strong></p>
                `;
                
                this.displayAssistantMessage(questionHtml);
            }

            displayClarifyingQuestions(questions) {
                const questionsHtml = `
                    <div class="step-indicator">Step 2</div>
                    <p>Great! I need a few more details to give you the best budget-filtered recommendations:</p>
                    <ul class="question-list">
                        ${questions.map((q, i) => `<li><strong>Q${i+1}:</strong> ${q}</li>`).join('')}
                    </ul>
                    <p>Let's start with the first question:</p>
                    <p><strong>${questions[0]}</strong></p>
                `;
                
                this.displayAssistantMessage(questionsHtml);
            }

            displayNextQuestion(question, questionIndex) {
                const questionHtml = `
                    <div class="step-indicator">Question ${questionIndex + 1}</div>
                    <p><strong>${question}</strong></p>
                `;
                this.displayAssistantMessage(questionHtml);
            }

            async startAnalysis() {
                this.displayAssistantMessage(
                    `<div class="step-indicator">Step 3</div>` +
                    `<div class="loading">Generating balanced recommendations for all your preferences</div>`
                );

                try {
                    const response = await fetch('/api/analyze_requirements', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: this.sessionId })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    if (data.type === 'recommendations') {
                        this.displayRecommendations(data.recommendations);
                    } else {
                        throw new Error('Unexpected response format');
                    }
                } catch (error) {
                    console.error('Analysis failed:', error);
                    this.displayErrorMessage(`Analysis failed: ${error.message}. Please try again.`);
                }
            }

            displayRecommendations(recommendations) {
                const recommendationsHtml = `
                    <div class="step-indicator">Step 4</div>
                    <p>Based on balanced analysis of community discussions, here are recommendations covering all your interests:</p>
                    <div class="selection-container">
                        <div class="selection-title">Select your top 3 preferred products:</div>
                        <div class="selection-grid">
                            ${recommendations.map((rec, i) => {
                                const parts = rec.split(' - ');
                                const title = parts[0].replace(/^\\d+\\.\\s*/, '');
                                const description = parts[1] || '';
                                return `
                                    <div class="selection-item" data-index="${i}">
                                        <div class="selection-item-number">${i+1}</div>
                                        <div class="selection-item-title">${title}</div>
                                        <div class="selection-item-description">${description.substring(0, 100)}...</div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                        <button class="confirm-button" id="confirmSelection" disabled>Confirm Selection (0/3)</button>
                    </div>
                `;
                
                this.displayAssistantMessage(recommendationsHtml);
                this.initializeProductSelection();
            }

            initializeProductSelection() {
                const selectionItems = document.querySelectorAll('.selection-item');
                const confirmButton = document.getElementById('confirmSelection');
                let selectedIndices = [];
                
                selectionItems.forEach(item => {
                    item.addEventListener('click', () => {
                        const index = parseInt(item.dataset.index);
                        const isSelected = item.classList.contains('selected');
                        
                        if (isSelected) {
                            item.classList.remove('selected');
                            selectedIndices = selectedIndices.filter(i => i !== index);
                        } else if (selectedIndices.length < 3) {
                            item.classList.add('selected');
                            selectedIndices.push(index);
                        }
                        
                        confirmButton.textContent = `Confirm Selection (${selectedIndices.length}/3)`;
                        confirmButton.disabled = selectedIndices.length !== 3;
                    });
                });
                
                confirmButton.addEventListener('click', async () => {
                    await this.handleProductSelection(selectedIndices);
                });
            }

            async handleProductSelection(selectedIndices) {
                this.displayAssistantMessage(
                    `<div class="step-indicator">Step 5</div>` +
                    `<div class="loading">Finding budget-compliant products from Indian e-commerce sites</div>`
                );

                try {
                    const response = await fetch('/api/select_products', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: this.sessionId,
                            selected_indices: selectedIndices
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }

                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    if (data.type === 'final_recommendations') {
                        this.displayFinalRecommendations(data.product_results);
                    } else {
                        throw new Error('Unexpected response format');
                    }
                } catch (error) {
                    console.error('Product selection failed:', error);
                    this.displayErrorMessage(`Product search failed: ${error.message}. Please try again.`);
                }
            }

            displayFinalRecommendations(productResults) {
                const budgetInfo = productResults[0]?.budget_info;
                
                const recommendationsHtml = `
                    <div class="step-indicator">Complete</div>
                    <div class="final-recommendations">
                        <h3>🎁 Your Gift Recommendations</h3>
                        ${productResults.map(category => `
                            <div class="recommendation-category">
                                <div class="category-title">${category.category}</div>
                                ${category.products.length > 0 ? category.products.map(product => `
                                    <div class="product-card">
                                        <div class="product-title">
                                            ${product.title}
                                            ${product.budget_check?.extracted_prices?.length > 0 ? 
                                              `<span style="background-color: #10b981; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-left: 8px;">₹${Math.min(...product.budget_check.extracted_prices)}</span>` : 
                                              '<span style="background-color: #0ea5e9; color: white; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-left: 8px;">Budget-Friendly</span>'
                                            }
                                        </div>
                                        <div class="product-description">${product.snippet}</div>
                                        <div style="margin-top: 8px;">
                                            <a href="${product.url}" class="buy-link" target="_blank">
                                                🛒 View on ${product.site?.charAt(0).toUpperCase() + product.site?.slice(1) || 'Site'}
                                            </a>
                                        </div>
                                    </div>
                                `).join('') : `
                                    <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 16px; text-align: center; color: #dc2626;">
                                        <p>😔 No products found within your budget on Indian e-commerce sites.</p>
                                        <p>💡 Try increasing your budget or searching for similar alternatives.</p>
                                    </div>
                                `}
                            </div>
                        `).join('')}
                    </div>
                `;
                
                this.displayAssistantMessage(recommendationsHtml);
                this.setInputEnabled(false);
            }

            displayUserMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message user';
                messageDiv.innerHTML = `
                    <div class="message-avatar">You</div>
                    <div class="message-content">${message}</div>
                `;
                this.chatContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }

            displayAssistantMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                messageDiv.innerHTML = `
                    <div class="message-avatar">AI</div>
                    <div class="message-content">${message}</div>
                `;
                this.chatContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }

            displayErrorMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message assistant';
                messageDiv.innerHTML = `
                    <div class="message-avatar">AI</div>
                    <div class="message-content">
                        <div class="error-message">${message}</div>
                    </div>
                `;
                this.chatContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }

            setLoading(loading) {
                this.sendButton.disabled = loading;
                this.messageInput.disabled = loading;
            }

            setInputEnabled(enabled) {
                this.sendButton.disabled = !enabled;
                this.messageInput.disabled = !enabled;
                if (!enabled) {
                    this.messageInput.placeholder = "Conversation completed. Refresh to start a new session.";
                }
            }

            scrollToBottom() {
                this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
            }
        }

        // Initialize the application
        document.addEventListener('DOMContentLoaded', () => {
            new GiftRecommendationUI();
        });
    </script>
</body>
</html>"""
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("🚀 Budget-Filtered Gift Recommendation System Starting...")
    print(f"📍 Server will be available on port: {port}")
    print("🔒 Using environment variables for API keys")
    print("⚖️ Bias prevention and balanced recommendations enabled")
    print("🇮🇳 INDIAN SITES ONLY - Amazon.in, Flipkart, Myntra, etc.")
    print("💰 STRICT BUDGET FILTERING ENABLED - Only shows products within budget")
    
    if debug_mode:
        print("🧪 Test your APIs first by visiting: http://localhost:5000/api/test_apis")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
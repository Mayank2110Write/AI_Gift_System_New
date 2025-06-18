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

    def generate_clarifying_questions(self, user_query: str) -> List[str]:
        """EXACT copy from your original code"""
        analysis_prompt = f"""
        Analyze the following user query for gift recommendations and generate 3-4 thoughtful clarifying questions to better understand their requirements.

        USER QUERY: "{user_query}"

        Based on this query, analyze what information is missing or unclear, and generate questions to help understand:
        1. The recipient's specific interests, hobbies, and preferences
        2. Budget range and spending comfort level
        3. The occasion and relationship context
        4. Any specific constraints or requirements (size, type, practical vs. sentimental, etc.)
        5. The recipient's age, gender, or other relevant demographics if not clear

        Generate exactly 3-4 specific, helpful questions that will gather the most important missing information.

        Format your response as:
        1. [Question about recipient's interests/preferences]
        2. [Question about budget/occasion]
        3. [Question about constraints/specifics]
        4. [Question about recipient details if needed]

        Make the questions conversational and easy to answer. Focus on gathering information that will lead to better gift recommendations.

        Examples:
        - "What's your budget range for this gift?"
        - "What are their main hobbies or interests?"
        - "What's the occasion for this gift?"
        - "Are you looking for something practical or more sentimental?"
        """

        try:
            print("🤖 Analyzing user query with GPT-4o...")

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

            print("✅ CLARIFYING QUESTIONS GENERATED:")
            for i, question in enumerate(questions, 1):
                print(f"  {i}. {question}")

            return questions

        except Exception as e:
            print(f"✗ Error generating clarifying questions: {e}")
            # Fallback questions
            fallback_questions = [
                "What's your budget range for this gift?",
                "What are their main interests or hobbies?",
                "What's the occasion for this gift?",
                "Are you looking for something practical or more sentimental?"
            ]
            print("✓ Using fallback questions")
            return fallback_questions

    def create_enhanced_query(self, original_query: str, clarifying_answers: Dict[str, str]) -> str:
        """EXACT copy from your original code"""
        print("\n=== CREATING ENHANCED SEARCH CONTEXT ===")

        # Prepare enhanced context
        enhanced_context = f"Original request: {original_query}\n\n"
        enhanced_context += "Additional details:\n"

        for answer_data in clarifying_answers.values():
            enhanced_context += f"- {answer_data['question']} {answer_data['answer']}\n"

        print("✓ Enhanced context created:")
        print(f"  Original query: {original_query[:50]}...")
        print(f"  Additional details: {len(clarifying_answers)} answers collected")

        return enhanced_context

    def optimize_search_query(self, enhanced_context: str) -> List[str]:
        """EXACT copy from your original code"""
        print("\n=== OPTIMIZING SEARCH QUERIES WITH GPT-4O ===")

        optimization_prompt = f"""
        Based on the user requirements for gift recommendations, generate 4-5 SHORT, targeted Google search queries that will find the most relevant Reddit gift recommendation discussions.

        USER REQUIREMENTS:
        {enhanced_context}

        Create SIMPLE, focused queries that target different aspects:
        1. Primary query: Core gift need + recipient
        2. Subreddit-specific: Target r/giftideas or r/gifts
        3. Interest-specific: Based on hobbies/interests mentioned
        4. Budget/occasion specific: If mentioned in requirements
        5. Demographic specific: Age/relationship based if relevant

        IMPORTANT GUIDELINES:
        - Each query should be 3-8 words maximum
        - Always start with "reddit" (not "site:reddit.com")
        - Always add 'India' in the generated queries to better filter out reddit posts specific to indian demographics
        - Use natural, conversational keywords
        - Focus on ONE main aspect per query
        - Include gift-related terms in each query
        - Use common Reddit terminology

        GOOD Examples:
        - "reddit gift ideas brother birthday"
        - "reddit giftideas movies documentaries"
        - "reddit birthday gifts sports lover"
        - "reddit brother gift recommendations"
        - "reddit gifts men under budget"

        Generate exactly 4-5 queries, one per line. Return only the queries, nothing else.
        """

        try:
            print("🤖 Sending enhanced context to GPT-4o for optimization...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": optimization_prompt}],
                max_tokens=150,
                temperature=0.3
            )

            queries_text = response.choices[0].message.content.strip()

            # Parse queries into a list
            queries = []
            for line in queries_text.split('\n'):
                line = line.strip()
                if line and line.startswith('reddit'):
                    # Clean up any formatting
                    query = line.replace('"', '').replace('- ', '').strip()
                    queries.append(query)

            print(f"✅ GPT-4O GENERATED {len(queries)} OPTIMIZED SEARCH QUERIES:")
            for i, query in enumerate(queries, 1):
                print(f"   {i}. {query}")

            return queries

        except Exception as e:
            print(f"✗ Error optimizing queries with GPT-4o: {e}")
            # Fallback queries based on enhanced context
            fallback_queries = [
                "reddit gift ideas brother birthday India",
                "reddit giftideas birthday gifts India",
                "reddit brother gift recommendations India",
                "reddit birthday present ideas men India"
            ]
            print(f"✓ Using fallback queries: {fallback_queries}")
            return fallback_queries

    def search_reddit_posts(self, query_list: List[str]) -> List[Dict[str, str]]:
        """EXACT copy from your original code with improved filtering"""
        print("\n=== SEARCHING FOR REDDIT POSTS ===")

        found_posts = []
        seen_urls = set()

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

        for i, query in enumerate(query_list, 1):
            print(f"\n--- Search {i}/{len(query_list)}: {query} ---")

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

                # Stop if we have enough high-quality results
                if len(found_posts) >= 8:
                    print("  Sufficient high-quality posts found, stopping search")
                    break

            except requests.exceptions.Timeout:
                print(f"  ✗ Timeout error with query '{query}'")
                continue
            except Exception as e:
                print(f"  ✗ Error with query '{query}': {e}")
                continue

        # Sort by relevance score and return top results
        found_posts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        final_posts = found_posts[:6]  # Top 6 most relevant

        print(f"\n✅ SEARCH COMPLETED - FOUND {len(final_posts)} RELEVANT REDDIT POSTS")
        return final_posts

    def extract_reddit_content(self, reddit_posts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """EXACT copy from your original code"""
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

    def analyze_for_product_recommendations(self, reddit_content: List[Dict[str, Any]], enhanced_context: str) -> List[str]:
        """EXACT copy from your original code"""
        print("\n=== ANALYZING CONTENT FOR PRODUCT RECOMMENDATIONS ===")

        if not reddit_content:
            print("❌ No Reddit content to analyze")
            return []

        # Prepare content for analysis with URL mapping
        analysis_text = f"USER REQUIREMENTS:\n{enhanced_context}\n\n"
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

        print(f"✓ Prepared content for analysis ({len(analysis_text)} characters)")

        # Create analysis prompt with URL references
        analysis_prompt = f"""
        Analyze the following Reddit discussions about gift recommendations and extract the top 10 most recommended products/items based on the user's specific requirements.

        {analysis_text}

        Based on the user requirements and the Reddit discussions above, please:

        1. Identify ALL products, items, or gift ideas mentioned in the posts and comments
        2. Analyze which products are most frequently recommended or highly upvoted
        3. Consider the context of the user's specific requirements (budget, interests, occasion, etc.)
        4. Provide the TOP 10 most suitable product recommendations that match the user's needs

        IMPORTANT: For each recommendation, please include the exact post reference (POST_1, POST_2, etc.) where this product was mentioned so I can provide the direct URL.

        Format your response as:
        1. [Product Name/Category] - [Brief explanation why this is recommended, including price range if mentioned and how it fits the user's specific needs] | Source: [POST_X from the analysis above]
        2. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        3. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        4. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        5. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        6. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        7. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        8. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        9. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]
        10. [Product Name/Category] - [Brief explanation...] | Source: [POST_X]

        Focus on practical, specific recommendations that were actually discussed in the Reddit threads and match the user's detailed requirements. Include a variety of gift types at different price points. Always reference the specific POST_X identifier for each recommendation.
        """

        try:
            print("🤖 Sending content to GPT-4o for analysis...")

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=1500,  # Increased tokens for 10 recommendations
                temperature=0.3
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

            print("✅ ANALYSIS COMPLETED")
            print(f"Generated {len(recommendations)} product recommendations")

            return recommendations

        except Exception as e:
            print(f"✗ Error analyzing content: {e}")
            return []

    def search_products_to_buy(self, selected_products: List[str], enhanced_context: str) -> List[Dict[str, Any]]:
        """AI-powered product search with intelligent filtering"""
        print("\n=== SEARCHING FOR PRODUCTS TO BUY (AI-POWERED) ===")

        all_product_results = []

        for product in selected_products:
            # Extract clean product name from recommendation
            product_name = product.split(' - ')[0] if ' - ' in product else product
            if '. ' in product_name:
                product_name = product_name.split('. ', 1)[1]
            
            # Remove any remaining formatting
            product_name = product_name.replace('**', '').strip()

            print(f"\n--- Searching for: {product_name} ---")

            # Step 1: Get raw search results
            raw_results = self._get_raw_search_results(product_name)
            
            if not raw_results:
                print(f"  ❌ No search results found for {product_name}")
                all_product_results.append({
                    'category': product_name,
                    'original_recommendation': product,
                    'products': []
                })
                continue

            print(f"  📋 Found {len(raw_results)} raw search results")

            # Step 2: Use AI to filter and rank results
            filtered_products = self._ai_filter_and_rank_products(product_name, raw_results)
            
            all_product_results.append({
                'category': product_name,
                'original_recommendation': product,
                'products': filtered_products[:3]  # Top 3 AI-selected products
            })

            print(f"  ✅ AI selected {len(filtered_products[:3])} high-quality products for {product_name}")

        return all_product_results

    def _get_raw_search_results(self, product_name: str) -> List[Dict[str, str]]:
        """Get raw search results without heavy filtering"""
        raw_results = []
        seen_urls = set()

        # Simple, focused search queries
        queries = [
            f"{product_name} buy online India",
            f"{product_name} India price shopping",
            f"best {product_name} online India",
            f"{product_name} India e-commerce store"
        ]

        for query in queries:
            if len(raw_results) >= 20:  # Get more results for AI to choose from
                break
                
            try:
                search_url = "https://www.googleapis.com/customsearch/v1"
                params = {
                    'key': self.GOOGLE_API_KEY,
                    'cx': self.SEARCH_ENGINE_ID,
                    'q': query,
                    'num': 10
                }

                response = requests.get(search_url, params=params, timeout=15)
                
                if response.status_code != 200:
                    continue

                search_results = response.json()
                
                if 'items' not in search_results:
                    continue

                for item in search_results['items']:
                    url = item['link']
                    
                    # Skip if already seen
                    if url in seen_urls:
                        continue
                    
                    # Very basic filtering - just exclude obvious non-shopping sites
                    if any(site in url.lower() for site in ['reddit.com', 'quora.com', 'youtube.com']):
                        continue
                    
                    raw_results.append({
                        'title': item['title'],
                        'url': url,
                        'snippet': item.get('snippet', ''),
                        'query_used': query
                    })
                    
                    seen_urls.add(url)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    ✗ Error with query '{query}': {e}")
                continue

        return raw_results

    def _ai_filter_and_rank_products(self, product_name: str, raw_results: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Use OpenAI to intelligently filter and rank product results"""
        print(f"  🤖 Using AI to analyze {len(raw_results)} search results...")

        # Prepare results for AI analysis
        results_text = f"PRODUCT SEARCH: {product_name}\n\n"
        results_text += "SEARCH RESULTS TO ANALYZE:\n"
        results_text += "=" * 50 + "\n"

        for i, result in enumerate(raw_results, 1):
            results_text += f"\nRESULT {i}:\n"
            results_text += f"Title: {result['title']}\n"
            results_text += f"URL: {result['url']}\n"
            results_text += f"Description: {result['snippet']}\n"
            results_text += "-" * 30 + "\n"

        # AI analysis prompt
        ai_prompt = f"""
        You are a smart product search assistant. Analyze the search results below and identify the BEST 5 results that are:

        1. **Actual product pages** where users can BUY the product (not help pages, settings, blogs, forums)
        2. **From legitimate e-commerce websites** (not social media, discussion forums, or random blogs)
        3. **Relevant to the product**: {product_name}
        4. **Diverse sources** - prefer results from different websites when possible
        5. **High quality** - proper product titles, clear descriptions, buyable products

        REJECT results that are:
        - Help/support/contact pages
        - Settings/account/login pages  
        - Blog posts or articles ABOUT the product
        - Social media posts
        - Forum discussions
        - News articles
        - Review compilation pages (unless they have buy links)
        - Category pages without specific products
        - Search result pages
        - Non-shopping websites

        {results_text}

        INSTRUCTIONS:
        1. Analyze each result carefully
        2. Select the TOP 5 results that best meet the criteria above
        3. Rank them by quality and relevance
        4. Prefer diverse websites (don't pick all from same site)

        Respond in this EXACT format:
        SELECTED_RESULTS:
        1. RESULT_X - [Brief reason why this is a good product page]
        2. RESULT_Y - [Brief reason why this is a good product page]  
        3. RESULT_Z - [Brief reason why this is a good product page]
        4. RESULT_A - [Brief reason why this is a good product page]
        5. RESULT_B - [Brief reason why this is a good product page]

        If fewer than 5 good results exist, only list the good ones. Only include RESULT_X numbers that correspond to actual results above.
        """

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": ai_prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content.strip()
            print(f"  ✅ AI analysis completed")
            
            # Parse AI response to extract selected results
            selected_products = self._parse_ai_selection(ai_response, raw_results)
            
            return selected_products
            
        except Exception as e:
            print(f"  ❌ AI analysis failed: {e}")
            # Fallback: return first 3 results if AI fails
            return raw_results[:3]

    def _parse_ai_selection(self, ai_response: str, raw_results: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse AI response and return selected products"""
        selected_products = []

        # Look for the SELECTED_RESULTS section
        if "SELECTED_RESULTS:" in ai_response:
            selection_text = ai_response.split("SELECTED_RESULTS:")[1]
            
            # Parse each selected result
            for line in selection_text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit() and 'RESULT_' in line:
                    try:
                        # Extract result number (e.g., "RESULT_3" -> 3)
                        result_part = line.split('RESULT_')[1].split(' -')[0]
                        result_index = int(result_part) - 1  # Convert to 0-based index
                        
                        if 0 <= result_index < len(raw_results):
                            selected_result = raw_results[result_index].copy()
                            
                            # Extract AI reasoning
                            if ' - ' in line:
                                reasoning = line.split(' - ', 1)[1]
                                selected_result['ai_reasoning'] = reasoning
                            
                            selected_result['ai_selected'] = True
                            selected_products.append(selected_result)
                            
                            print(f"    ✅ AI selected: {selected_result['title'][:50]}...")
                            
                    except (ValueError, IndexError) as e:
                        print(f"    ⚠️ Error parsing AI selection line: {line}")
                        continue

        # If no valid selections found, return top 3 raw results as fallback
        if not selected_products:
            print("    ⚠️ AI selection parsing failed, using fallback")
            selected_products = raw_results[:3]

        return selected_products

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
        'message': 'Welcome! Please describe what kind of gift you\'re looking for and for whom.'
    })

@app.route('/api/process_message', methods=['POST'])
def process_message():
    """Process user messages using REAL Python functions"""
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        if session_data['step'] == 1:
            # Initial query - USE REAL FUNCTION
            session_data['user_query'] = message
            
            # REAL API CALL - not mock
            questions = gift_system.generate_clarifying_questions(message)
            session_data['clarifying_questions'] = questions
            session_data['step'] = 2
            
            return jsonify({
                'type': 'clarifying_questions',
                'questions': questions,
                'current_question': 0
            })
            
        elif session_data['step'] == 2:
            # Clarifying answers
            current_q_index = len(session_data['clarifying_answers'])
            session_data['clarifying_answers'][f'question_{current_q_index + 1}'] = {
                'question': session_data['clarifying_questions'][current_q_index],
                'answer': message
            }
            
            if current_q_index + 1 < len(session_data['clarifying_questions']):
                # More questions to ask
                return jsonify({
                    'type': 'next_question',
                    'question': session_data['clarifying_questions'][current_q_index + 1],
                    'current_question': current_q_index + 1
                })
            else:
                # All questions answered, start analysis
                session_data['step'] = 3
                return jsonify({
                    'type': 'start_analysis',
                    'message': 'Processing your preferences...'
                })
        
        else:
            return jsonify({'error': 'Invalid step'}), 400
            
    except Exception as e:
        print(f"Error processing message: {e}")
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/api/analyze_requirements', methods=['POST'])
def analyze_requirements():
    """Analyze user requirements using REAL Python functions - NO MOCK DATA"""
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        print("\n🚀 STARTING REAL ANALYSIS - NO MOCK DATA")
        
        # STEP 1: Create enhanced context using REAL function
        enhanced_context = gift_system.create_enhanced_query(
            session_data['user_query'], 
            session_data['clarifying_answers']
        )
        
        # STEP 2: Generate optimized search queries using REAL function
        print("📋 Generating optimized search queries...")
        optimized_queries = gift_system.optimize_search_query(enhanced_context)
        
        if not optimized_queries:
            raise Exception("Failed to generate search queries")
        
        # STEP 3: Search Reddit posts using REAL function
        print("🔍 Searching Reddit posts...")
        reddit_posts = gift_system.search_reddit_posts(optimized_queries)
        
        if not reddit_posts:
            raise Exception("No Reddit posts found")
        
        # STEP 4: Extract Reddit content using REAL function
        print("📖 Extracting Reddit content...")
        reddit_content = gift_system.extract_reddit_content(reddit_posts)
        
        if not reddit_content:
            raise Exception("No Reddit content extracted")
        
        # STEP 5: Analyze for product recommendations using REAL function
        print("🤖 Analyzing content for recommendations...")
        recommendations = gift_system.analyze_for_product_recommendations(reddit_content, enhanced_context)
        
        if not recommendations:
            raise Exception("No recommendations generated")
        
        # Clean up ** stars from product names
        cleaned_recommendations = []
        for rec in recommendations:
            # Remove ** from product names/titles
            cleaned_rec = rec.replace('**', '')
            cleaned_recommendations.append(cleaned_rec)
        
        # Store recommendations in session
        session_data['recommendations'] = cleaned_recommendations
        session_data['reddit_content'] = reddit_content  # Store for later use
        session_data['enhanced_context'] = enhanced_context  # Store for later use
        session_data['step'] = 4
        
        print(f"✅ REAL ANALYSIS COMPLETED - {len(cleaned_recommendations)} recommendations generated")
        
        return jsonify({
            'type': 'recommendations',
            'recommendations': cleaned_recommendations
        })
        
    except Exception as e:
        print(f"❌ Error in real analysis: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/select_products', methods=['POST'])
def select_products():
    """Handle product selection using REAL Python functions - NO MOCK DATA"""
    data = request.json
    session_id = data.get('session_id')
    selected_indices = data.get('selected_indices')
    
    if session_id not in conversation_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    session_data = conversation_sessions[session_id]
    
    try:
        print(f"\n🎯 PROCESSING REAL PRODUCT SELECTION - Indices: {selected_indices}")
        
        # Get selected products from stored recommendations
        selected_products = [session_data['recommendations'][i] for i in selected_indices]
        session_data['selected_products'] = selected_products
        
        print("✅ Selected products:")
        for i, product in enumerate(selected_products, 1):
            print(f"  {i}. {product[:100]}...")
        
        # REAL PRODUCT SEARCH using your original function
        print("🛒 Searching for actual products to buy...")
        product_results = gift_system.search_products_to_buy(
            selected_products, 
            session_data['enhanced_context']
        )
        
        print(f"✅ REAL PRODUCT SEARCH COMPLETED - Found results for {len(product_results)} categories")
        
        return jsonify({
            'type': 'final_recommendations',
            'product_results': product_results
        })
        
    except Exception as e:
        print(f"❌ Error in real product selection: {e}")
        return jsonify({'error': f'Product selection failed: {str(e)}'}), 500

# Add a test endpoint to verify API connectivity
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Save the HTML template (your original HTML content)
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
            <p>Get personalized gift recommendations from community discussions</p>
        </div>

        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-avatar">AI</div>
                <div class="message-content">
                    <div class="step-indicator">Step 1</div>
                    <p>Welcome! I'll help you find the perfect gift by analyzing community discussions and finding specific products to buy.</p>
                    <p>Please describe what kind of gift you're looking for and for whom:</p>
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
                    case 'clarifying_questions':
                        this.displayClarifyingQuestions(data.questions);
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

            displayClarifyingQuestions(questions) {
                const questionsHtml = `
                    <div class="step-indicator">Step 2</div>
                    <p>Great! I need a few more details to give you the best recommendations:</p>
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
                    <div class="step-indicator">Question ${questionIndex + 1}/${questionIndex + 2}</div>
                    <p><strong>${question}</strong></p>
                `;
                this.displayAssistantMessage(questionHtml);
            }

            async startAnalysis() {
                this.displayAssistantMessage(
                    `<div class="step-indicator">Step 3</div>` +
                    `<div class="loading">Generating personalized recommendations</div>`
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
                    <p>Based on community discussions, here are the top recommendations:</p>
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
                    `<div class="loading">Finding products for purchase</div>`
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
                const recommendationsHtml = `
                    <div class="step-indicator">Complete</div>
                    <div class="final-recommendations">
                        <h3>Your Personalized Gift Recommendations - Ready to Buy!</h3>
                        ${productResults.map(category => `
                            <div class="recommendation-category">
                                <div class="category-title">${category.category}</div>
                                ${category.products.length > 0 ? category.products.map(product => `
                                    <div class="product-card">
                                        <div class="product-title">${product.title}</div>
                                        <div class="product-description">${product.snippet}</div>
                                        <a href="${product.url}" class="buy-link" target="_blank">View Product</a>
                                    </div>
                                `).join('') : '<p>No products found for this category.</p>'}
                            </div>
                        `).join('')}
                    </div>
                    <p>System completed successfully! You now have specific products with purchase links for your selected gift categories.</p>
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
    
    print("🚀 Secure Gift Recommendation System Starting...")
    print(f"📍 Server will be available on port: {port}")
    print("🔒 Using environment variables for API keys")
    
    if debug_mode:
        print("🧪 Test your APIs first by visiting: http://localhost:5000/api/test_apis")
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
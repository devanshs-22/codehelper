CodeHelper: Custom Modifications Log
This document outlines the series of modifications made to the original interview-coder-withoupaywall-opensource repository to enhance its functionality, add support for newer AI models, and package it as a standalone macOS application.

1. Added Support for New Gemini Models
The initial goal was to add support for more advanced models like Gemini 2.5 Pro and Gemini 1.5 Flash. This was a two-step process.

Step 1.1: Updating the User Interface

First, we added the new models to the selection dropdown in the settings menu so they could be chosen by the user.

File Modified: SettingsDialog.tsx

Change: Added new model objects to the geminiModels array within the modelCategories constant.


Step 1.2: Updating the Backend Validation

Next, we had to teach the application's backend to recognize and accept these new model IDs when saving the configuration.

File Modified: electron/ConfigHelper.ts

Change: Added the new model IDs to the allowedModels array inside the sanitizeModelSelection function.

Improving API Request Robustness
We discovered that larger models like Gemini 2.5 Pro could sometimes fail on complex tasks due to short timeouts or insufficient token limits. We made the application more robust to prevent these failures.

File Modified: electron/ProcessingHelper.ts

Changes:

Increased Timeout: The API call timeout was universally increased to 3 minutes (180,000 ms) for all major API calls, giving the models ample time to think.

Increased Token Limit: The maxOutputTokens for Gemini solution generation was increased to 50000 to allow for longer, more detailed code responses.


 Advanced Prompt Engineering for Versatility
To make the app more versatile, we implemented two major prompt engineering improvements.

Step 3.1: Adding MCQ Solving Capability

The app was originally designed only for coding problems. We implemented a robust two-step process to handle both coding problems and Multiple-Choice Questions (MCQs) accurately.

File Modified: electron/ProcessingHelper.ts

Change:

Classification Step: A preliminary API call was added to a fast model (gemini-1.5-flash-latest) to classify the user's screenshot as either 'coding' or 'mcq'.

Conditional Logic: An if/else block was added. If the request is an MCQ, it uses a dedicated prompt to get the correct answer and explanation. If it's a coding problem, it proceeds with the original logic.

Step 3.2: Optimizing Prompts for Higher Quality Output

We fine-tuned the prompts to improve the quality of the output for both coding and debugging tasks.

File Modified: electron/ProcessingHelper.ts

Changes:

Solution Prompt: The prompt was updated to instruct the AI to act as a "skilled software engineer," focusing on clean, efficient, and standard optimal solutions rather than overthinking the problem.

Debug Prompt: The prompt was changed to instruct the AI to provide the complete, corrected code first, followed by a brief explanation of the fixes, prioritizing the solution over the analysis.

4. Building a Standalone macOS Application
To transform the project from a development script into a professional, standalone application, we configured it for a production build.

Step 4.1: Installing the Builder

Action: Installed electron-builder, the standard tool for packaging Electron applications.

Command: npm install electron-builder --save-dev

Step 4.2: Configuring the Build

We modified the package.json to create a disguised, standalone .dmg installer for macOS.

File Modified: package.json

Changes:

Added a "build" script to the "scripts" section.

Added a comprehensive "build" configuration block to define the application's final properties.

Disguised Name: Set productName to "CoreAudioHelper" to appear as a system process in Activity Monitor.

Disguised Icon: Configured the build to use a custom icon located at assets/icons/mac/icon.icns. We chose a generic speaker icon to match the product name.

Build Target: Set the target to dmg for easy installation on macOS.

Step 4.3: Building the Application

Action: Ran the final build command to generate the standalone application.

Command: npm run package-mac

Result: A CoreAudioHelper.dmg file was created in the release folder, ready for distribution and installation.

5. Repository Management
To take ownership of the project and secure our modifications, we detached it from the original public repository and created a new private one.

Action:

Created a new private repository on GitHub named codehelper.

Removed the old Git history from the local folder with rm -rf .git.

Initialized a new Git repository, committed all our changes, and pushed them to the new private codehelper remote repository.








/**
 * English Language Pack (Minimal - Dynamic Content Only)
 * Original HTML text is preserved, this pack only covers JS-generated content
 */

const en = {
    // Status messages
    status: {
        connected: 'Connected',
        disconnected: 'Disconnected',
        connecting: 'Connecting...',
        serverRunning: 'Server Running',
        serverStopped: 'Server Stopped',
        serverStarting: 'Server Starting...',
        offline: 'Offline',
        online: 'Online'
    },
    
    // Server messages
    server: {
        starting: 'Starting vLLM server...',
        stopping: 'Stopping vLLM server...',
        started: 'Server started successfully',
        stopped: 'Server stopped',
        error: 'Server error',
        ready: 'Server is ready',
        notReady: 'Server is not ready'
    },
    
    // Chat messages
    chat: {
        thinking: 'Thinking...',
        generating: 'Generating response...',
        stopped: 'Generation stopped',
        error: 'Error generating response',
        send: 'Send',
        clear: 'Clear Chat',
        clearConfirm: 'Are you sure you want to clear all chat history?'
    },
    
    // Log messages
    log: {
        connected: 'WebSocket connected',
        disconnected: 'WebSocket disconnected',
        error: 'Error',
        warning: 'Warning',
        info: 'Info',
        success: 'Success'
    },
    
    // Validation messages
    validation: {
        required: 'This field is required',
        invalidPath: 'Invalid path',
        pathNotFound: 'Path not found',
        validating: 'Validating...',
        valid: 'Valid',
        invalid: 'Invalid'
    },
    
    // Benchmark messages
    benchmark: {
        running: 'Benchmark running...',
        completed: 'Benchmark completed',
        failed: 'Benchmark failed',
        starting: 'Starting benchmark...',
        stopping: 'Stopping benchmark...'
    },
    
    // Tool messages
    tool: {
        added: 'Tool added',
        updated: 'Tool updated',
        deleted: 'Tool deleted',
        error: 'Tool error',
        calling: 'Calling tool...',
        executionResult: 'Execution Result'
    },
    
    // File operations
    file: {
        uploading: 'Uploading...',
        uploaded: 'File uploaded',
        uploadError: 'Upload error',
        downloading: 'Downloading...',
        downloaded: 'Downloaded'
    },
    
    // Common actions
    action: {
        save: 'Save',
        cancel: 'Cancel',
        delete: 'Delete',
        edit: 'Edit',
        add: 'Add',
        remove: 'Remove',
        confirm: 'Confirm',
        close: 'Close',
        reset: 'Reset',
        apply: 'Apply',
        browse: 'Browse',
        search: 'Search',
        clear: 'Clear',
        copy: 'Copy',
        paste: 'Paste',
        start: 'Start',
        stop: 'Stop'
    },
    
    // Error messages
    error: {
        unknown: 'Unknown error occurred',
        network: 'Network error',
        timeout: 'Request timeout',
        serverError: 'Server error',
        invalidInput: 'Invalid input',
        notFound: 'Not found',
        forbidden: 'Access forbidden',
        unauthorized: 'Unauthorized'
    },
    
    // Time-related
    time: {
        justNow: 'Just now',
        minutesAgo: '{{minutes}} minutes ago',
        hoursAgo: '{{hours}} hours ago',
        daysAgo: '{{days}} days ago',
        uptime: 'Uptime: {{time}}'
    },
    
    // Units
    units: {
        tokens: 'tokens',
        seconds: 'seconds',
        minutes: 'minutes',
        hours: 'hours',
        mb: 'MB',
        gb: 'GB'
    },
    
    // Theme
    theme: {
        toggle: 'Toggle dark/light mode',
        dark: 'Dark',
        light: 'Light'
    }
};

// Register language pack
if (window.i18n) {
    window.i18n.register('en', en);
}

